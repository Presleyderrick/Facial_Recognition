import face_recognition
import cv2
import os
import numpy as np

known_faces_folder = "known_faces"

# RTSP stream URL of the camera
rtsp_url = "rtsp://admin:Kap5_adm!n@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"

# Get a reference to the RTSP stream
video_capture = cv2.VideoCapture(rtsp_url)

# Check if the video stream is accessible
if not video_capture.isOpened():
    print("Error: Unable to access the RTSP stream")
    exit()

# Arrays of known face encodings and names
known_face_encodings = []
known_face_names = []

# Check if the 'known_faces' folder exists
if not os.path.exists(known_faces_folder):
    print(f"Error: Folder '{known_faces_folder}' does not exist.")
    exit()

# Iterate through each image file in the 'known_faces' folder
for filename in os.listdir(known_faces_folder):
    file_path = os.path.join(known_faces_folder, filename)

    # Only process image files (you can modify this to filter specific file types)
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            # Load the image and extract the face encoding
            image = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(image)

            # Check if a face was detected
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                # The name of the person is the filename without the extension
                name = os.path.splitext(filename)[0]

                # Append the encoding and name to the lists
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            else:
                print(f"No face found in image: {filename}")

        except Exception as e:
            print(f"Error processing image {filename}: {e}")

# Check if any known faces were loaded
if len(known_face_encodings) == 0:
    print("No faces were found in the known_faces folder.")
else:
    print(f"Loaded {len(known_face_encodings)} known faces.")

# Now `known_face_encodings` and `known_face_names` co

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video from RTSP stream
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Only process every other frame to save time
    if process_this_frame:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert to RGB (required by face_recognition library)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Find face encodings for all detected faces
        face_encodings = []
        for face_location in face_locations:
            # Get face landmarks for each face location
            landmarks = face_recognition.face_landmarks(rgb_small_frame, [face_location])
            
            # Ensure landmarks were detected before encoding
            if landmarks:
                # Convert landmarks to _dlib_pybind11.full_object_detection
                encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])[0]
                face_encodings.append(encoding)

        face_names = []
        for face_encoding in face_encodings:
            # Match the current face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the best match based on face distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back to original resolution
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the result in a window
    cv2.imshow('Video', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture handle
video_capture.release()
cv2.destroyAllWindows()
