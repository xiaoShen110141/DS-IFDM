var ws = new WebSocket("ws://127.0.0.1:4041/");
ws.onopen = function () {
    ws.send(document.location.href);
    //send a message to server once ws is opened.
    console.log("%c开始检测网页"+document.location.href+"%c出现的图片...","font-size:1.5em");
};

ws.onmessage = function (event) {
    if (typeof event.data === "string") {
        // If the server has sent text data, then display it.
        obj = event.data;
        var str = "";
        for (var i=0;i<obj.length-1;i++){
            if (obj[i]=='-' && obj[i]=='>'){
                console.log(str)
                str = ""
            }
            else{
                str = str+obj[i]
            } 
        }
        }
};

ws.onerror   = function (error) {
    console.log('Error Logged: ' + error); //log errors
};
