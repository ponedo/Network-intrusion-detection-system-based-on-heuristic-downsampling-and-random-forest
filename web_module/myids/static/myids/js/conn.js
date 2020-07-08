$(function() {
    loadConnData(condition, 1, page_size);
    loadConnCols();
});


function loadConnData(condition, page_id, page_size) {
    $.ajax({
        url: page_data_url + 
            "?page_id=" + page_id +
            "&req=data" +
            "&condition=" + condition + 
            "&page_size=" + page_size +
            "&from_time=" + start_time +
            "&to_time=" + end_time, 
        type: "get", 
        data: null, 
        success: function(resp){
            var tbody = $("#mainbody-table-body");
            var resp_data = resp["data"];

            /* Update table */
            tbody.empty();
            for (var i = 0; i < resp_data.length; i++) {
                var tr = $("<tr></tr>").text("");
                var row_data = resp_data[i];
                for (var j = 0; j < row_data.length; j++) {
                    var td = $("<td></td>").text(row_data[j]);
                    tr.append(td);
                }
                tbody.append(tr);
            }

            /* Update pagin button */
            var resp_count = resp["count"];
            var page_num = Math.ceil(resp_count / page_size);
            function pageLoad(page_id) {
                loadConnData(condition, page_id, page_size);
            }
            pagination($("#mainbody-pagination"), page_id, page_num, pageLoad);
            
        }, 
        error: function() {
            alert("error!!!");
        }
    });
}

function loadConnCols() {
    $.ajax({
        url: page_data_url + 
            "?req=cols", 
        type: "get", 
        data: null, 
        success: function(resp){
            var cols = resp['cols'];
            var colSelect = $(".select-cols");
            colSelect.empty();
            for (var i = 0; i < cols.length; i++) {
                var opt = $("<option></option").text(cols[i]);
                colSelect.append(opt);
            }
        }, 
        error: function() {
            alert("error!!!");
        }
    });
}
