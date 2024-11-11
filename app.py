import cv2
import serial
import face_recognition
from face_recognition_utils import load_known_faces
from serial_comm import setup_serial, send_serial_data

def main():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()
    if not known_face_encodings:
        print("No known faces found. Please add faces using capture_face.py.")
        return

    # Setup serial connection
    try:
        serial_conn = setup_serial()
    except serial.SerialException as e:
        print(e)
        serial_conn = None

    # RTSP stream URL
    rtsp_url = "rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
    video_capture = cv2.VideoCapture(rtsp_url)

    if not video_capture.isOpened():
        print("Error: Unable to access the RTSP stream")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]

            face_names.append(name)

            if serial_conn:
                send_serial_data(serial_conn, name)

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
