from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 3)  # Output classes: Healthy, Mild, Severe
model.load_state_dict(torch.load("resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define recommendations
recommendations = {
    "Healthy": "No action needed. Maintain good farming practices.",
    "Mild": "Apply organic fungicides and remove affected leaves.",
    "Severe": "Use systemic fungicides immediately and improve airflow."
}

# Flask route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_names = ["Healthy", "Mild", "Severe"]
    label = class_names[predicted_class]
    response = {
        "class": label,
        "recommendation": recommendations[label],
        "confidence": probabilities.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)
