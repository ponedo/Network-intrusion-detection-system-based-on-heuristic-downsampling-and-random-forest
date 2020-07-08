from django.db import connection


def get_distinct():
    """
        Get the number of proto types.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT distinct proto FROM conn LIMIT 10;"
        )
        distinct = cursor.fetchall()
        distinct = tuple(rec[0] for rec in distinct)
        return distinct


def get_count(from_time=None, to_time=None, specific=None):
    """
        Get the count of each proto type.
    """
    if from_time and to_time:
        time_condition_sql = "ts between '{}' and '{}' ".format(from_time, to_time)
    else:
        time_condition_sql = ""
    if specific:
        specific = ["'" + rec + "'" if isinstance(rec, str) else str(rec) for rec in specific]
        arr = ", ".join(specific)
        specific_sql = "proto IN ({}) ".format(arr)
    else:
        specific_sql = ""
    if time_condition_sql and specific_sql:
        condition_sql = "WHERE " + time_condition_sql + "AND " + specific_sql
    elif time_condition_sql and not specific_sql:
        condition_sql = "WHERE " + time_condition_sql
    elif not time_condition_sql and specific_sql:
        condition_sql = "WHERE " + specific_sql
    else:
        condition_sql = " "
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT proto, count(*) as cnt FROM conn " + \
            condition_sql + \
            "GROUP BY proto " + \
            "ORDER BY cnt DESC " + \
            "LIMIT 10;"
        )
        count = cursor.fetchall()
        return count


def get_stat_by(page_size=10, page_id=None, from_time=None, to_time=None):
    """
        Get the count of 
        alert_num, src_ip_num, src_port_num, dest_ip_num, dest_port_num, service_num
        for each proto type.
    """
    if from_time and to_time:
        time_condition_sql = "WHERE ts between '{}' and '{}' ".format(from_time, to_time)
    else:
        time_condition_sql = ""
    with connection.cursor() as cursor:
        count = cursor.execute(
            "SELECT proto, \
            count(*) as cnt, \
            count(distinct alert) as unique_alert_num, \
            count(distinct orig_h) as unique_src_ip_num, \
            count(distinct orig_p) as unique_src_port_num, \
            count(distinct dest_h) as unique_dest_ip_num, \
            count(distinct dest_p) as unique_dest_port_num, \
            count(distinct service) as unique_service_num \
            FROM conn " + \
            time_condition_sql + \
            "GROUP BY proto " + \
            "ORDER BY cnt DESC;"
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


def get_specific_stat_by(proto, stat, page_size=10, page_id=None, from_time=None, to_time=None):
    """
        Get the count and timestamp of distinct stat:stat 
        for records with proto type of param:proto
    """
    if stat == "src_ip":
        stat = "orig_h"
    elif stat == "src_port":
        stat = "orig_p"
    elif stat == "dest_ip":
        stat = "dest_h"
    elif stat == "dest_port":
        stat = "dest_p"
    if from_time and to_time:
        time_condition_sql = "AND ts between '{}' and '{}' ".format(from_time, to_time)
    else:
        time_condition_sql = "  "
    sql = """
        SELECT """ + stat + """, 
        count(*) as count, 
        min(ts) as earliest_occurence, 
        max(ts) as last_occurence 
        FROM conn 
        WHERE proto = %s 
        """ + time_condition_sql + """ 
        GROUP BY """ + stat + """
        ORDER BY count DESC;
    """
    with connection.cursor() as cursor:
        count = cursor.execute(sql, proto)
        if page_id and page_size:
            cursor.scroll((page_id-1) * page_size, mode='absolute')
            data = cursor.fetchmany(page_size)
        else:
            data = cursor.fetchall()
        return {
            "data": data, 
            "count": count
        }
