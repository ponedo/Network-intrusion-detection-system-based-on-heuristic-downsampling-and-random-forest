function loadPie(containerId, dataType) {
    $.ajax({
        url: query_stat_url + 
            "?requester=pie" + 
            "&data_type=" + dataType +
            "&from_time=" + start_time +
            "&to_time=" + end_time, 
        type: "get", 
        data: null, 
        success: function(resp) {
            renderPieChart(containerId, resp);
        }, 
        error: function() {}
    });
}


function loadLine(containerId, dataType) {
    $.ajax({
        url: query_stat_url + 
            "?requester=line" +
            "&data_type=" + dataType +
            "&from_time=" + start_time +
            "&to_time=" + end_time, 
        type: "get", 
        data: null, 
        success: function(resp) {
            renderLineChart(containerId, resp);
        }, 
        error: function() {}
    });
}


function loadStatOverview(container, dataType, page_id, page_size) {
    $.ajax({
        url: query_stat_url + 
            "?requester=stat_overview" +
            "&data_type=" + dataType +
            "&from_time=" + start_time +
            "&to_time=" + end_time +
            "&page_id=" + page_id +
            "&page_size=" + page_size, 
        type: "get", 
        data: null, 
        success: function(resp) {
            function pageLoad(page_id) {
                loadStatOverview(container, dataType, page_id, page_size);
            }
            renderStatOverview(container, dataType, resp, page_id, page_size, pageLoad);
        }, 
        error: function() {}
    });
}


function loadStatDetail(container, dataType, dataValue, statType, page_id, page_size) {
    $.ajax({
        url: query_stat_url + 
            "?requester=stat_detail" +
            "&data_type=" + dataType +
            "&data_value=" + dataValue +
            "&stat_type=" + statType +
            "&from_time=" + start_time +
            "&to_time=" + end_time +
            "&page_id=" + page_id +
            "&page_size=" + page_size, 
        type: "get", 
        data: null, 
        success: function(resp) {
            function pageLoad(page_id) {
                loadStatDetail(container, dataType, dataValue, statType, page_id, page_size);
            }
            renderStatDetail(container, dataType, resp, page_id, page_size, pageLoad);
        }, 
        error: function() {}
    });
}