var mainInput = document.querySelectorAll("#mainInput");
$(document).ready(function(){
    
    ${'#mainInput'}.keyup(function(e){
        if (e.keyCode == 32) {
            // console.log('press space key to fatch data');
            predictionAPI(e);
            return;
        } 
        
        function predictionAPI(event){
            var texttosend = mainInput[0].value
            texttosend = texttosend.slice(0, texttosend.length-1)
            $.ajax({
                url: "/predict",
                type: "POST",
                data: {text:texttosend}}).done(function(response){
                    $("#text1").html("I");
                    $("#text2").html(response.predicted.predicted);
                    $("#text3").html("am");
                });
        }
    });

    $("#text1").click(function(){
        var new_text = mainInput[0].value + $("#text1").text();
        mainInput[0].value =  new_text;
    }); 

    $("#text2").click(function(){
        var new_text = mainInput[0].value + $("#text2").text();
        mainInput[0].value =  new_text;
    }); 

    $("#text3").click(function(){
        var new_text = mainInput[0].value + $("#text3").text();
        mainInput[0].value =  new_text;
    }); 

    


});

