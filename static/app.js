document.addEventListener("DOMContentLoaded", function () {
    const chatButton = document.querySelector(".chatbox__button");
    const chatBox = document.querySelector(".chatbox__support");

    // Toggle chatbox visibility
    chatButton.addEventListener("click", function () {
        if (chatBox.style.display === "none" || chatBox.style.display === "") {
            chatBox.style.display = "flex"; // Open the chatbox
        } else {
            chatBox.style.display = "none"; // Close the chatbox
        }
    });

    const sendButton = document.querySelector(".send__button");
    const chatInput = document.getElementById("chatInput");
    const chatMessages = document.getElementById("chatboxMessages");

    sendButton.addEventListener("click", function () {
        sendMessage();
    });

    chatInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = chatInput.value.trim();
        if (userMessage === "") return;

        // Append user message (Right Side)
        const userMessageDiv = document.createElement("div");
        userMessageDiv.classList.add("messages__item", "messages__item--operator");
        userMessageDiv.innerText = userMessage;
        chatMessages.appendChild(userMessageDiv);
        chatInput.value = "";

        fetch("/chatbot", {
            method: "POST",
            body: JSON.stringify({ message: userMessage }),
            headers: { "Content-Type": "application/json" }
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.response;

            // Append chatbot response (Left Side)
            const botMessageDiv = document.createElement("div");
            botMessageDiv.classList.add("messages__item", "messages__item--visitor");
            botMessageDiv.innerText = botResponse;
            chatMessages.appendChild(botMessageDiv);

            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => console.error("Error:", error));
    }
});
