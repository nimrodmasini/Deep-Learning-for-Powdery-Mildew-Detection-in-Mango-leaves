import os
import json
import torch
import requests
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 3) 
model.load_state_dict(torch.load("../resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Recommendations based on severity
recommendations = {
    "Healthy": "To maintain this healthy condition, it's important to continue regular monitoring of your plant. Good practices include ensuring proper air circulation between trees by pruning, avoiding overhead irrigation which can increase humidity, and applying preventive organic treatments like neem oil or sulfur-based sprays occasionally. Staying consistent with these care routines helps prevent disease before it starts.",
    "Mild": " The disease is present but not yet widespread. Quick intervention can stop it from spreading further. Start by carefully removing the affected leaves if they are few and localized. Then, apply a mild organic fungicide such as neem oil or a potassium bicarbonate solution. Improving air circulation by pruning nearby foliage and avoiding overhead watering can also help reduce humidity, which encourages mildew growth. Monitor the plant closely over the next few days to ensure the infection does not progress.",
    "Severe": "Immediate action is needed to protect the plant and nearby crops. Begin by pruning and safely disposing of heavily infected leaves or branches to reduce the fungal load. Follow this with an application of a strong systemic fungicide approved for agricultural use, such as one containing myclobutanil. Avoid watering the leaves directly and aim to keep the plant as dry as possible. It's also advisable to consult a local agricultural expert or extension officer for further assessment and guidance, especially if the infection has spread to multiple plants."
}

#API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  

@csrf_exempt
def classify_image(request):
    """Classifies an uploaded image and returns disease severity and recommendation."""
    if request.method == 'POST' and request.FILES.get('image'):
        image = Image.open(request.FILES['image'])
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
        return JsonResponse(response)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def inquiry(request):
    """Handles AI-based user inquiries related to disease severity using DeepSeek."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            disease_class = data.get('disease_class', '').strip()

            if not question:
                return JsonResponse({'error': 'No question provided'}, status=400)

            prompt = (
                f"Provide a concise answer about {disease_class} powdery mildew in mango plants. "
                f"Question: {question}\n"
                "Format your response with no markdown and make them short and clear."
            )

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {"role": "system", "content": "You are an agricultural expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7  
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()

            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({
                "answer": f"**Error**: Could not process your request. ({str(e)})"
            }, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def home(request):
    """Renders the homepage."""
    return render(request, 'index.html')
