/* data selector */
$(function () {
    var curDate = new Date();
    var curYear = curDate.getFullYear();

    function isLeapYear(year) {
        return (year % 4 == 0 && year % 100 != 0);
    }

    function createOptions(element, n) {
        element.empty();
        for (var i = 1; i <= n; i++) {
            var v;
            if (i < 10)
                v = '0' + i;
            else
                v = '' + i;
            var newOpt = $("<option></option").text(v);
            element.append(newOpt);
        }
    }

    for (var i = curYear-5; i < curYear+5; i++) {
        var newOpt = $("<option></option").text(i);
        newOpt.val(i);
        $(".select-year").append(newOpt);
    }
    createOptions($(".select-month"), 12);
    createOptions($(".select-day"), 31);

    $(".select-year").change(function() {
        var yearSelVal = Number($(this).val());
        var monSelVal = Number($(this).parent().find(".select-month").val());
        var daySel = $(this).parent().find(".select-day");
        if (monSelVal == 2) {
            if (isLeapYear(yearSelVal))
                createOptions(daySel, 29);
            else
                createOptions(daySel, 28);
        }
    });

    $(".select-month").change(function() {
        var yearSelVal = Number($(this).parent().find(".select-year").val());
        var monSelVal = Number($(this).val());
        var daySel = $(this).parent().find(".select-day");
        switch (monSelVal) {
            case 1:
            case 3:
            case 5:
            case 7:
            case 8:
            case 10:
            case 12:
                createOptions(daySel, 31);
                break;
            case 4:
            case 6:
            case 9:
            case 11:
                createOptions(daySel, 30);
                break;
            case 2:
                if (isLeapYear(yearSelVal))
                    createOptions(daySel, 29);
                else
                    createOptions(daySel, 28);
                break;
        }
    });
    
});

/* time selector */
window.onload = function() {
    var hours = document.getElementsByClassName("select-hour");
    for (var j = 0; j < hours.length; j++) {
        hour = hours[j];
        for (var i = 0; i < 24; i++) {
            var opt = document.createElement("option");
            // opt.value = i;
            if (i < 10)
                opt.innerHTML = "0" + i.toString();
            else
                opt.innerHTML = i;
            hour.appendChild(opt);
        }
    };
    var mins = document.getElementsByClassName("select-minute");
    for (var j = 0; j < mins.length; j++) {
        min = mins[j];
        for (var i = 0; i < 60; i++) {
            var opt = document.createElement("option");
            // opt.value = i;
            if (i < 10)
                opt.innerHTML = "0" + i;
            else
                opt.innerHTML = i;
            min.appendChild(opt);
        }
    };
    var secs = document.getElementsByClassName("select-second");
    for (var j = 0; j < secs.length; j++) {
        sec = secs[j];
        for (var i = 0; i < 60; i++) {
            var opt = document.createElement("option");
            // opt.value = i;
            if (i < 10)
                opt.innerHTML = "0" + i;
            else
                opt.innerHTML = i;
            sec.appendChild(opt);
        }
    };
}