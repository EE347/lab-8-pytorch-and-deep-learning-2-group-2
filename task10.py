import cv2
import torch
import numpy as np
from picamera2 import Picamera2
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet_v3_small(num_classes=2).to(device)
model.load_state_dict(torch.load('lab8/best_model_CrossEntropy.pth', map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Initialize the Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Load OpenCV’s pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
assert not face_cascade.empty(), "Failed to load the face cascade classifier XML file."

# Labels for classification
class_labels = ["Teammate 0", "Teammate 1"]

def classify_face(image):
    """Process image for model prediction, returning classification result."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Main loop for live video stream
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop and process each detected face
            face = frame[y:y+h, x:x+w]
            if face.shape[0] >= 64 and face.shape[1] >= 64:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB for model input
                face_resized = cv2.resize(face_rgb, (64, 64))  # Resize to model's expected input size

                # Classify the face and get the label
                prediction = classify_face(face_resized)
                label = class_labels[prediction]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the live video stream with annotations
        cv2.imshow('Teammate Classification - Live Stream', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()