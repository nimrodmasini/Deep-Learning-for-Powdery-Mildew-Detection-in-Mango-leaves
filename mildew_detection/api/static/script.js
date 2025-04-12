// Function to handle image classification
function classifyImage() {
    // Get the selected file from the input element
    const input = document.getElementById('imageInput').files[0];

    // Check if an image was selected
    if (!input) {
        alert("Please select an image first.");
        return;
    }

    // Create a FileReader to preview the image
    const reader = new FileReader();
    reader.onload = function (e) {
        const preview = document.getElementById("previewImage");
        preview.src = e.target.result;  // Set the preview image source
        preview.style.display = "block";  // Show the preview image
    };
    reader.readAsDataURL(input);  // Read the image file as a data URL

    // Prepare image data for sending to the backend
    const formData = new FormData();
    formData.append("image", input);

    // Send image to the backend for classification
    fetch("/api/classify/", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())  // Parse the JSON response
    .then(data => {
        // Display the classification result and recommendation
        document.getElementById("result").innerHTML = `
            <h2>🧪 Result: ${data.class}</h2>
            <p><strong>Recommendation:</strong> ${data.recommendation}</p>
            <textarea id="inquiry" placeholder="Ask more about this..."></textarea>
            <button id="askBtn" onclick="askLLM('${data.class}')">Ask AI</button>
            <div id="aiResponse"></div>`;
    })
    .catch(err => {
        // Handle errors and display an error message
        console.error("Error:", err);
        document.getElementById("result").innerHTML = "Error analyzing image.";
    });
}

// Function to handle AI-based inquiries
function askLLM(diseaseClass) {
    // Get the user’s question from the textarea
    const question = document.getElementById("inquiry").value.trim();

    // Check if the question is empty
    if (!question) {
        alert("Please enter your question.");
        return;
    }

    // Send the question and disease class to the backend
    fetch("/api/inquiry/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, disease_class: diseaseClass })
    })
    .then(res => res.json())  // Parse the JSON response
    .then(data => {
        // Parse and display the AI-generated response using Markdown
        const markdown = marked.parse(data.answer);
        document.getElementById("aiResponse").innerHTML = `
            <div class="response-content">${markdown}</div>`;
    })
    .catch(err => {
        // Handle errors and display an error message
        console.error("Error:", err);
        document.getElementById("aiResponse").innerHTML = "Error fetching AI response.";
    });
}
