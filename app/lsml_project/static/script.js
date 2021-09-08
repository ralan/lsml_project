$(document).ready(function () {
    function check_status(task_id) {
        $.ajax({
            url: "/api/predict/" + task_id,
            type: "get",
            success: function (data) {
                if (data["status"] == "DONE") {
                    $("#answer").text("Predicted answer: " + data["answer"])
                    $("#predict-answer-button").prop('disabled', false);
                    $("#spinner").addClass("d-none");
                    $("#answer").removeClass("d-none");
                }
                else {
                    setTimeout(function () {
                        check_status(task_id)
                    }, 1000);
                }
            }
        })
    }

    $("#predict-answer-button").on("click", function (event) {
        $("#predict-answer-button").prop('disabled', true);
        $("#answer").addClass("d-none");
        $("#spinner").removeClass("d-none");

        $.ajax({
            url: "/api/predict",
            type: "post",
            data: JSON.stringify({
                question: $("#question").val(),
                passage: $("#passage").val()
            }),
            contentType: "application/json; charset=utf-8",
            success: function (data) {
                check_status(data["task_id"])
            }
        })
    });
});