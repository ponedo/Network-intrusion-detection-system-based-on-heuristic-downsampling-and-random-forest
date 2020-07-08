function renderPieChart(containerId, resp) {
    $("#" + containerId).empty();
    Highcharts.chart(containerId, {
        chart: {
            styledMode: true, 
            spacingLeft: -150, 
            spacingRight: 100,
            marginRight: 0, 
        },
        title: {
            text: null, 
        },
        legend: {
        　　align: 'right', 
        　　verticalAlign: 'middle', 
            layout: 'vertical'
        }, 
        credits:{
            enabled: false
        }, 
        series: [{
            type: 'pie',
            allowPointSelect: true,
            keys: ['name', 'y', 'selected', 'sliced'],
            data: resp["series"], 
            showInLegend: true, 
            dataLabels: {
                enabled: false
            }, 
        }], 
    });
}


function renderLineChart(containerId, resp) {
    $("#" + containerId).empty();
    Highcharts.chart(containerId, {
        title: {
            text: null
        },
        xAxis: {
            categories: resp["xAxis"]
        },
        yAxis: {
            title: {
                text: 'count'
            }
        },
        legend: {
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle', 
        },
        credits:{
            enabled: false
        }, 
        plotOptions: {
            series: {
                label: {
                    connectorAllowed: false
                },
            }
        },
        series: resp["series"],
    });
}


function renderStatOverview(container, dataType, resp, page_id, page_size, pageLoad) {
    var thead = container.find("thead");
    var tbody = container.find("tbody");
    var cols = resp["cols"];
    var data = resp["data"];
    var trueDataType;
    switch (dataType) {
        case "src_ip": trueDataType = "orig_h"; break;
        case "src_port": trueDataType = "orig_p"; break;
        case "dest_ip": trueDataType = "dest_h"; break;
        case "dest_port": trueDataType = "dest_p"; break;
        default: trueDataType = dataType; break;
    }

    /* Render stat overview to table */
    thead.empty();
    tbody.empty();
    for (var j = 0; j < cols.length; j++) {
        var th = $("<th></th>").text(cols[j]);
        thead.append(th);
    }
    for (var i = 0; i < data.length; i++) {
        var tr = $("<tr></tr>").text("");
        var row_data = data[i];
        for (var j = 0; j < row_data.length; j++) {
            var td = $("<td></td>").text("");
            if (j == 1) {
                var dataValue = row_data[0];
                if (typeof dataValue == "string") {
                    dataValue = "@" + dataValue + "@";
                }
                var condition = trueDataType + " = " + dataValue;
                var a = $("<a>", {
                    text: row_data[j], 
                    href: conn_url + 
                        "?condition=" + condition +
                        "&from_time=" + start_time +
                        "&to_time=" + end_time
                });
                td.append(a);
            } else { 
                var span = $("<span></span>").text(row_data[j]);
                if (j > 1) {
                    span.css({
                        "cursor": "pointer", 
                        "color": "blue"
                    });
                    span.click(
                        {dataType: dataType}, 
                        overviewJumpToDetail
                    );
                }
                td.append(span);
            }
            tr.append(td);
        }
        tbody.append(tr);
    }
    
    var paginElement = container.parent().find(".pagination");
    var resp_count = resp["count"];
    var page_num = Math.ceil(resp_count / page_size);
    pagination(paginElement, page_id, page_num, pageLoad);
}


function renderStatDetail(container, dataType, resp, page_id, page_size, pageLoad) {
    var thead = container.find("thead");
    var tbody = container.find("tbody");
    var cols = resp["cols"];
    var data = resp["data"];
    thead.empty();
    tbody.empty();
    for (var j = 0; j < cols.length; j++) {
        var th = $("<th></th>").text(cols[j]);
        thead.append(th);
    }
    for (var i = 0; i < data.length; i++) {
        var tr = $("<tr></tr>").text("");
        var row_data = data[i];
        for (var j = 0; j < row_data.length; j++) {
            var td = $("<td></td>").text(row_data[j]);
            tr.append(td);
        }
        tbody.append(tr);
    }
    
    var paginElement = container.parent().find(".pagination");
    var resp_count = resp["count"];
    var page_num = Math.ceil(resp_count / page_size);
    pagination(paginElement, page_id, page_num, pageLoad);
}


function overviewJumpToDetail(event) {
    var overviewPanel = $("#stat-overview-panel");
    var overviewTable = $("#stat-overview-table");
    var detailPanel = $("#stat-detail-panel");
    var detailTable = $("#stat-detail-table");

    var dataType = event.data.dataType;
    var siblings = $(this).parent().parent().children();
    var dataValue = siblings.first().text();
    var colId = siblings.index($(this).parent());
    var statType = overviewTable.find("th").eq(colId).text();

    overviewPanel.hide();
    detailPanel.show();
    loadStatDetail(
        detailTable, dataType, dataValue, statType, 1, detailPageSize
    );

    var detailTitle = detailPanel.find("#stat-detail-title");
    var titleText = "Detailed " + statType + " statistics for " + dataType + " = " + dataValue;
    detailTitle.text(titleText);
}