function pagination(paginElement, page_id, page_num, loadData) {
    paginElement.empty();
    paginElement.css({
        "text-align": "center", 
        "padding": "10px 0", 
    });

    /* Add buttons */
    for (var i = 1; i <= page_num; i++) {
        var pagin_button = $('<span>', {
            text: i,
            class: "pagin-button"
        });
        if (i == page_id) {                        
            pagin_button.attr('disabled', 'disabled');
            pagin_button.css({
                "color": "white", 
                "background-color": "skyblue", 
                "cursor": "default"
            });
            paginElement.append(pagin_button);
        } else if (
            i == 1 || i == 2 || 
            i == page_num-1 || i == page_num ||
            i == page_id-1 || i == page_id+1) {
            paginElement.append(pagin_button);
        } else if (i == page_id-2 || i == page_id+2) {
            paginElement.append($("<span></span>").text("..."));
        } else {
            continue;
        }
    }

    /* Add goto text input */
    var inputText = $("<input>", {
        text: "", 
        class: "pagin-input"
    });
    var gotoButton = $("<button>", {
        text: "Go", 
        class: "pagin-jump-button"
    });
    paginElement.append(inputText);
    paginElement.append(gotoButton);


    /* Add event listener */
    $(".pagin-button").click(function () {
        var page_id = Number($(this).text());
        loadData(page_id);
    });
    gotoButton.click(function() {
        var page_id = Number($(this).parent().find("input").val());
        if (page_id <= page_num && page_id > 0) {
            loadData(page_id);
        } else {
            alert("Requested page is illegal or out of range.");
        }
    });
    
}