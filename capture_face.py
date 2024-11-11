import cv2
import face_recognition
import numpy as np
import os

ENCODINGS_PATH = "data/encodings.npy"
NAMES_PATH = "data/names.npy"

# Initialize the lists to store known face encodings and names
known_face_encodings = np.empty((0, 128))  # Initialize as an empty 2D array (0 rows, 128 columns)
known_face_names = []

# Load previously saved encodings and names if they exist
if not os.path.exists('data'):
    os.makedirs('data')

if os.path.exists(ENCODINGS_PATH) and os.path.exists(NAMES_PATH):
    known_face_encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
    known_face_names = np.load(NAMES_PATH, allow_pickle=True)
    print(f"Loaded {len(known_face_names)} known faces.")

# RTSP URL for the video feed
rtsp_url = "rtsp://admin:Kap5_adm!n@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
video_capture = cv2.VideoCapture(rtsp_url)

# Check if the video stream is opened successfully
if not video_capture.isOpened():
    print("Error: Unable to access the RTSP stream")
    exit()

# Variables to store face locations, encodings, and names
face_locations = []
face_encodings = []
face_names = []

# Flag to control frame processing speed
process_this_frame = True

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture frame.")
        break

    # Only process every other frame to save processing time
    if process_this_frame:
        # Convert the frame to RGB (face_recognition uses RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any of the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If there is a match, use the corresponding name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                # If no match, prompt to add a new face
                name = input("Enter name for this new face: ")

                # Append the new face encoding to the list of known encodings
                known_face_encodings = np.append(known_face_encodings, [face_encoding], axis=0)
                known_face_names.append(name)

            # Print the name of the recognized face (optional)
            print(f"Recognized face: {name}")

    # Toggle frame processing to reduce the computational load
    process_this_frame = not process_this_frame

# Save the updated face encodings and names
np.save(ENCODINGS_PATH, known_face_encodings)
np.save(NAMES_PATH, known_face_names)

# Release the video capture object when done
video_capture.release()
print("Face capture completed.")
