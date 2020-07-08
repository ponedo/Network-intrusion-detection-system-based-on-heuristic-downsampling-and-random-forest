$(function() {
    $(".add-condition-button").hide();
    $(".add-conjunction-button").hide();
    $(".condition-container").hide();

    $(".panel-control-button").click(function() {
        if($(".condition-container").is(':hidden')){
            $(".condition-container").show();
        } else {
            $(".condition-container").hide();
        }
    });
    

    $(".add-atomic-button").click(function() {
        /* Obtain submitted values */
        var col = $(".select-cols").val();
        var op = $(".select-operator").val();
        var cmpVal = $(".cmp-value-input").val();
        if (cmpVal == "") {
            alert("Compared value can't be empty!");
            return;
        }
        var cmpValNum = Number(cmpVal);
        if (isNaN(cmpValNum))
            cmpVal = "'" + cmpVal + "'";
        
        /* Add atomic */
        var addToPanel = $(".added-atomic");
        var txt;
        if (addToPanel.children().length == 0) {
            txt = col + " " + op + " " + cmpVal;
        } else {
            txt = " AND " + col + " " + op + " " + cmpVal;
        }
        var renderSpan = $("<span>", {
            class: "conjunction-span",
            text: txt
        })
        addToPanel.append(renderSpan);
        addToPanel.append($("<br>").text(""));
        $(".add-conjunction-button").show();
        
    })

    $(".add-conjunction-button").click(function() {
        var inputPanel = $(".added-atomic");
        var addToPanel = $(".added-conjunction");
        var conjunction = "";

        /* Get atomic */
        var atomicElements = inputPanel.children();
        for (var i = 0; i < atomicElements.length; i++) {
            var tagName = atomicElements[i].tagName;
            if (!(tagName == "SPAN")) 
                continue;
            conjunction += atomicElements.eq(i).text();
        }
        inputPanel.empty();

        /* Add conjunction */
        var txt;
        if (addToPanel.children().length == 0) {
            txt = conjunction;
        } else {
            txt = " OR " + conjunction;
        }
        var renderSpan = $("<span>", {
            class: "conjunction-span",
            text: txt
        })
        addToPanel.append(renderSpan);
        addToPanel.append($("<br>").text(""));
        $(".add-condition-button").show();
        $(this).hide();
    })

    $(".add-condition-button").click(function() {
        var inputPanel = $(".added-conjunction");
        var queryCondition = "";

        /* Get conjunctions */
        var conjunctionElements = inputPanel.children();
        for (var i = 0; i < conjunctionElements.length; i++) {
            var tagName = conjunctionElements[i].tagName;
            if (!(tagName == "SPAN")) 
                continue;
                queryCondition += conjunctionElements.eq(i).text();
        }
        inputPanel.empty();

        /* Generate query condition and query*/
        var msg = "Your query condition is as follow: \n" + 
            queryCondition + "\n" + 
            "Sure to query?"
        if (confirm(msg)) {
            $(".add-conjunction-button").hide();
            $(".added-atomic").hide();
            $(this).hide();
            loadConnData(queryCondition, 1, page_size);
            $(".condition-container").hide();
        }
    })
});