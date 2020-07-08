from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.template import loader
import datetime
import json
import importlib

# Create your views here.

##################
# Page rendering #
##################
def view_index(request):
    context = {}
    data_types = ['alert', 'proto', 'src_ip', 'dest_ip', 'src_port', 'dest_port']
    time_context = set_time(request)
    context.update(time_context)
    context['data_types'] = data_types
    return render(request, 'myids/index.html', context)


def view_stat(request):
    context = {}
    data_types = ['alert', 'proto', 'src_ip', 'dest_ip', 'src_port', 'dest_port']
    time_context = set_time(request)
    context.update(time_context)
    context['data_types'] = data_types
    return render(request, 'myids/stat.html', context)


def view_conn(request):
    condition = request.GET.get("condition", "null")
    context = {
        "condition": condition, 
    }
    time_context = set_time(request)
    context.update(time_context)
    return render(request, 'myids/conn.html', context)


#########################
# Response without page #
#########################
@csrf_exempt
def query_conn(request):
    if request.GET:
        req = request.GET['req']
        conn = importlib.import_module(".dbutil.conn", package="myids")
        if req == "data":
            condition = request.GET['condition']
            condition = condition.replace('@', "'")
            condition = None if condition == "null" else condition
            page_id = int(request.GET['page_id'])
            page_size = int(request.GET['page_size'])
            from_time = request.GET['from_time']
            to_time = request.GET['to_time']
            data_and_count = conn.get(page_id=page_id, page_size=page_size, from_time=from_time, to_time=to_time, condition=condition)
            resp_content = {}
            resp_content.update(data_and_count)
        elif req == "cols":
            cols = conn.get_cols()
            cols = [rec[0] for rec in cols]
            resp_content = {
                "cols": cols
            }
        return JsonResponse(
                resp_content, 
                encoder=JsonCustomEncoder, 
                status='200')
    else:
        return HttpResponse(content="error", charset='utf-8', status='404')
    

def query_stat(request):
    if request.GET:
        form = request.GET
        data_type = form["data_type"]
        requester = form["requester"]
        from_time, to_time = form.get('from_time'), form.get('to_time')
        db_module = importlib.import_module(".dbutil." + data_type, package="myids")

        if requester == "pie":
            data = db_module.get_count(from_time=from_time, to_time=to_time)
            resp_content = {
                "series": data
            }

        elif requester == "line":
            xAxis = []
            distinct = db_module.get_count(from_time=from_time, to_time=to_time)
            distinct = tuple(rec[0] for rec in distinct)
            raw_data = {}
            for entry in distinct:
                raw_data[entry] = []
            from_day, to_day = from_time[:10], to_time[:10]
            t = datetime.datetime.strptime(from_day + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_t = datetime.datetime.strptime(to_day + " 23:59:59", "%Y-%m-%d %H:%M:%S")
            one_day = datetime.timedelta(days=1)
            while t < end_t:
                xAxis_time = t.strftime("%m-%d")
                xAxis.append(xAxis_time)
                query_from_time = t.strftime("%Y-%m-%d %H:%M:%S")
                t += one_day
                query_to_time = t.strftime("%Y-%m-%d %H:%M:%S")
                day_data = db_module.get_count(from_time=query_from_time, to_time=query_to_time, specific=distinct)
                unrecorded_keys = list(distinct)
                for record in day_data:
                    entry, count = record[0], record[1]
                    raw_data[entry].append(count)
                    unrecorded_keys.remove(entry)
                for entry in unrecorded_keys:
                    raw_data[entry].append(0)
            data = []
            for name, data_series in raw_data.items():
                data.append({'name': name, 'data': data_series})
            resp_content = {
                "series": data, 
                "xAxis": xAxis,
            }

        elif requester == "stat_overview":
            page_id, page_size = int(form.get("page_id")), int(form.get("page_size"))
            data_and_count = db_module.get_stat_by(page_size=page_size, page_id=page_id, from_time=from_time, to_time=to_time)
            cols = ["conn count", "alert", "proto", "src_ip", "src_port", "dest_ip", "dest_port", "service"]
            cols.remove(data_type)
            cols[0:0] = [data_type]
            resp_content = {
                "cols": cols
            }
            resp_content.update(data_and_count)

        elif requester == "stat_detail":
            condition_value = form.get("data_value")
            stat_type = form.get("stat_type")
            page_id, page_size = int(form.get("page_id")), int(form.get("page_size"))
            data_and_count = db_module.get_specific_stat_by(
                condition_value, stat_type, page_size=page_size, page_id=page_id, from_time=from_time, to_time=to_time
            )
            cols = [stat_type, "count", "earlieat occurence", "last occurence"]
            resp_content = {
                "cols": cols
            }
            resp_content.update(data_and_count)

        else:
            data = None
            resp_content = {
                "data": data
            }
        
        return JsonResponse(
                resp_content, 
                encoder=JsonCustomEncoder, 
                status='200')
    else:
        return HttpResponse(content="error", charset='utf-8', status='404')


###########
# Helpers #
###########
class JsonCustomEncoder(json.JSONEncoder):
    def default(self, field):
        if isinstance(field, datetime.datetime):
            return field.strftime('%Y-%m-%d %H:%M:%S')
        # elif isinstance(field, date):
        #     return field.strftime('%Y-%m-%d')
        elif isinstance(field, bytes):
            if field == b'0':
                return 0
            elif field == b'1':
                return 1 
        else:
            return list(json.JSONEncoder.default(self, field))


def set_time(request):
    if (request.POST):
        p = request.POST
        start_time = p["from-year"] + '-' + p["from-month"] + '-' + p["from-day"] + \
            ' ' + p["from-hour"] + ':' + p["from-minute"] + ':' + p["from-second"]
        end_time = p["to-year"] + '-' + p["to-month"] + '-' + p["to-day"] + \
            ' ' + p["to-hour"] + ':' + p["to-minute"] + ':' + p["to-second"]
    elif request.GET:
        start_time = request.GET.get("from_time", "null")
        end_time = request.GET.get("to_time", "null")
    else:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=30)
        start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    context = {
        'start_time': start_time, 
        'end_time': end_time, 
    }
    return context