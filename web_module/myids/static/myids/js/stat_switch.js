$(function() {
    var selected = $("#alert-switch");
    selected.css('background-color', 'lightblue');
    selected.attr('disabled', 'disabled');

    $(".mainbody-switch-item").click(function() {
        //change css of switch buttons
        selected.css('background-color', 'white');
        selected.removeAttr('disabled');
        selected = $(this);
        selected.css('background-color', 'lightblue');
        selected.attr('disabled', 'disabled');

        //reload data
        var dataType = $(this).text();
        $("#stat-detail-panel").hide();
        loadPie("stat-pie-container", dataType);
        loadLine("stat-line-container", dataType);
        loadStatOverview($("#stat-overview-table"), dataType, 1, overviewPageSize);

        //reload title
        var title = dataType + " overview" + " (Charts show only top 10)";
        title = title.charAt(0).toUpperCase() + title.slice(1);
        $("#stat-overview-title").text(title);
    })

    $(".mainbody-switch-item").mouseover(function() {
        $(this).css("background-color", "cornflowerblue");
    })

    $(".mainbody-switch-item").mouseout(function() {
        if ($(this).is(selected)){
            $(this).css("background-color", "lightblue");
        }
        else
            $(this).css("background-color", "white");
    })
});