function sendMessage() {
    var userInput = document.getElementById("user-input");
    var message = userInput.value.trim();
    if (message !== "") {
      displayUserMessage(message);
      userInput.value = "";
      fetch('/send_message', {
        method: 'POST',
        body: new URLSearchParams({
          'message': message
        }),
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      })
      .then(response => response.json())
      .then(data => displayBotMessage(data.message))
      .catch(error => console.error('Error:', error));
    }
  }
  
  function displayUserMessage(message) {
    var chatMessages = document.getElementById("chat-messages");
    var userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.textContent = message;
    chatMessages.appendChild(userMessage);
  }
  
  function displayBotMessage(message) {
    var chatMessages = document.getElementById("chat-messages");
    var botMessage = document.createElement("div");
    botMessage.className = "bot-message";
    botMessage.textContent = message;
    chatMessages.appendChild(botMessage);
  }
  