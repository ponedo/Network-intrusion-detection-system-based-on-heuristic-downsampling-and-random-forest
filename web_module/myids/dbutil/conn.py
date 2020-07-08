from django.db import connection

def get_cols():
    sql = """
        SELECT COLUMN_NAME from information_schema.COLUMNS
        WHERE table_name = 'conn';
    """
    with connection.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()
        return data


def get(page_size=10, page_id=None, from_time=None, to_time=None, condition=None):
    if from_time and to_time:
        time_condition_sql = "ts between '{}' and '{}' ".format(from_time, to_time)
    else:
        time_condition_sql = ""
    if condition:
        general_condition_sql = condition + " "
    else:
        general_condition_sql = ""
    if time_condition_sql and general_condition_sql:
        condition_sql = "WHERE " + time_condition_sql + "AND " + general_condition_sql
    elif not time_condition_sql and general_condition_sql:
        condition_sql = "WHERE " + general_condition_sql
    elif time_condition_sql and not general_condition_sql:
        condition_sql = "WHERE " + time_condition_sql
    elif not time_condition_sql and not general_condition_sql:
        condition_sql =  " "
    
    with connection.cursor() as cursor:
        count = cursor.execute(
            "SELECT ts, proto, orig_h, orig_p, \
            dest_h, dest_p, service, duration, \
            orig_bytes, resp_bytes, orig_pkts, \
            resp_pkts, alert FROM conn " +\
            condition_sql + \
            "ORDER BY ts;"
        )
        if page_id and page_size:
            cursor.scroll((page_id-1) * page_size, mode='absolute')
            data = cursor.fetchmany(page_size)
        else:
            data = cursor.fetchall()
        return {
            "data": data, 
            "count": count
        }
