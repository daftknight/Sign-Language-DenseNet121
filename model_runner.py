import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features

# Load the state dict
checkpoint = torch.load('best_model.pth', map_location=device)
num_classes = checkpoint['model_state_dict']['classifier.weight'].size(0)
model.classifier = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Define the classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Ensure the classes list matches the number of classes in the model
if len(classes) != num_classes:
    raise ValueError(f"Number of classes in the model ({num_classes}) does not match the length of the classes list ({len(classes)}).")

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Convert the frame back to BGR (3 channels)
    processed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR)
    
    # Convert the frame to PIL image
    pil_image = Image.fromarray(processed_frame)
    
    # Apply the transformation
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = classes[predicted.item()]
    
    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_class} ({confidence.item():.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Prediction', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()