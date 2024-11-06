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

def classify_face(image):
    """Process image for model prediction, returning classification result."""
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    except Exception as e:
        print("Error during classification:", e)
        return None

# Main loop to capture and classify
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected")

        for (x, y, w, h) in faces:
            # Crop the face region from the frame
            face = frame[y:y+h, x:x+w]

            # Debugging: Print the shape of the detected face
            print("Detected face shape:", face.shape)

            # Resize and classify the face if it meets minimum size criteria
            if face.shape[0] >= 64 and face.shape[1] >= 64:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB for model input
                face_resized = cv2.resize(face_rgb, (64, 64))  # Resize to model's expected input size

                # Debugging: Print the shape after resizing
                print("Resized face shape:", face_resized.shape)

                prediction = classify_face(face_resized)

                if prediction is not None:
                    # Overlay classification result on the frame
                    label = "Teammate 0" if prediction == 0 else "Teammate 1"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    print("Classification failed for detected face.")
            else:
                print("Face detected but too small for classification")

        # Show the frame with annotations
        cv2.imshow('Teammate Classification', frame)

        # Break with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()