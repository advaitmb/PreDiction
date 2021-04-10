
var suggestionText = document.querySelectorAll("#suggestionText");
var userText = document.querySelectorAll("#userText");

let userTextHandler = {
    userTextSelector : document.querySelectorAll("#userText"),
    log : function () {
        console.log(this.userTextSelector[0].value)
    }
}
console.log("Something")

$("#userText").keyup(function(event){
    console.log("Hello there")
    userTextHandler.log();  
})
