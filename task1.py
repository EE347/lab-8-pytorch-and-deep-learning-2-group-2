import cv2
import os
from picamera2 import Picamera2
import time

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up directories
directories = ['data/train/0', 'data/train/1', 'data/test/0', 'data/test/1']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Give the camera time to warm up
time.sleep(2)

# Number of images to capture
images_per_person = 60
train_images = 50
test_images = 10

# Loop over each person (assuming two people, 0 and 1)
for person_id in [0, 1]:
    captured_count = 0
    print(f"Capturing images for person {person_id}...")

    while captured_count < images_per_person:
        # Capture a frame from Picamera2
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each face found
        for (x, y, w, h) in faces:
            # Crop and resize face to 64x64 pixels
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))

            # Determine whether to save in train or test folder
            if captured_count < train_images:
                folder = f'data/train/{person_id}'
            else:
                folder = f'data/test/{person_id}'

            # Save the face image with a unique name
            image_path = f"{folder}/{captured_count}.jpg"
            cv2.imwrite(image_path, face_resized)
            print(f"Saved image {image_path}")

            captured_count += 1

            # Stop once we've captured the required number of images
            if captured_count >= images_per_person:
                break

        # Display the frame with detected face for feedback
        cv2.imshow("Capturing Face", frame)

        # Press 'q' to stop early if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Completed capturing for person {person_id}.")

    # Wait for 'c' key to continue to the next person
    if person_id == 0:
        print("Press 'c' to start capturing images for the next person.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
print("All done!")
