import cv2
from simple_facerec import SimpleFacerec

# Encode faces from the folder
sfr = SimpleFacerec()
sfr.load_encoding_images("Images/")

# Load Camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect Faces
        face_locs, face_names = sfr.detect_known_faces(frame)

        # Draw rectangles around faces and label names
        for face_loc, name in zip(face_locs, face_names):
            top, right, bottom, left = face_loc
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Label the face with a name
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break

except KeyboardInterrupt:
    print("Program interrupted manually")

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
