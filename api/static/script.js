$(document).ready(function(){
    var mainInput = document.querySelectorAll("#mainInput");
    let input_length = 0;
    console.log(fullUrl)
    $('#mainInput').keyup(function(e){ 

        // console.log('press space key to fatch data');
        if (e.keyCode == 32) {
            input_length = mainInput[0].value.length;
            predictionAPI();
            return;
        }
        else {
            var value = mainInput[0].value
            var val = value.substr(input_length, value.length)
            // for (i = 0; i < arr.length; i++) {
            //     /*check if the item starts with the same letters as the text field value:*/
            //     if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
            }  
    });

    $("#text1").click(function(){
        var new_text = $("#text1").text();
        mainInput[0].value +=  new_text;
        $('#mainInput').focus().val($('#mainInput').val() + " ");
        predictionAPI();
    }); 

    $("#text2").click(function(){
        var new_text = $("#text2").text();
        mainInput[0].value += new_text;
        $('#mainInput').focus().val($('#mainInput').val() + " ");
        predictionAPI();
    }); 

    $("#text3").click(function(){
        var new_text = $("#text3").text();
        mainInput[0].value +=  new_text;
        $('#mainInput').focus().val($('#mainInput').val() + " ");
        predictionAPI(e);
    }); 

    function predictionAPI(event){
            var texttosend = mainInput[0].value
            texttosend = texttosend.slice(0, texttosend.length-1)
            $.ajax({
                url: "/predict",
                type: "POST",
                data: {text:texttosend}}).done(function(response){
                    $("#text1").html(response.predicted.word2);
                    $("#text2").html(response.predicted.word1);
                    $("#text3").html(response.predicted.word3);
                });
        }
});
