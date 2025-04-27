document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    let userInput = document.getElementById("user-input");
    let chatBox = document.getElementById("chat-box");
    
    let message = userInput.value.trim();
    if (!message) {
        alert("Type something!");
        return;
    }

    // Append user message
    let userMessageDiv = document.createElement("div");
    userMessageDiv.className = "message user-message";
    userMessageDiv.innerHTML = `<b>You:</b> ${message}`;
    chatBox.appendChild(userMessageDiv);

    userInput.value = "";

    // Append AI typing indicator (loading animation)
    let aiMessageDiv = document.createElement("div");
    aiMessageDiv.className = "message ai-message";
    aiMessageDiv.innerHTML = `<b>AI:</b> <span class="dots">...</span>`;
    chatBox.appendChild(aiMessageDiv);

    // Auto-scroll
    chatBox.scrollTop = chatBox.scrollHeight;

    // Fetch AI response
    let response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    });

    let data = await response.json();

    // Remove typing animation and add AI response
    aiMessageDiv.innerHTML = `<b>AI:</b> ${data.response}`;

    // Auto-scroll
    chatBox.scrollTop = chatBox.scrollHeight;
}
