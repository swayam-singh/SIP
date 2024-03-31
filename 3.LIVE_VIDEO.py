import cv2
import dlib

# Load the face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the Haar cascade classifiers for the smile and eyes
eyes_cascade = cv2.CascadeClassifier("opencvlib/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencvlib/haarcascade_smile.xml")

# Create VideoCapture object to capture video from webcam
cap = cv2.VideoCapture(0)

# Loop until user quits
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame was successfully captured
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using dlib
        faces = detector(gray)

        # Loop through each detected face
        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract coordinates of landmarks for mouth, eyes, and brows
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
            mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
            brow_left = (landmarks.part(17).x, landmarks.part(17).y)
            brow_right = (landmarks.part(26).x, landmarks.part(26).y)

            # Calculate distances between mouth corners and eyes for mouth width
            mouth_width = abs(mouth_right[0] - mouth_left[0])
            eye_distance = abs(right_eye[0] - left_eye[0])

            # Calculate ratio of mouth width to eye distance
            mouth_eye_ratio = mouth_width / eye_distance

            # If the mouth-eye ratio is within a certain range, consider it as relaxed facial features
            if 0.5 < mouth_eye_ratio < 1.0:
                # Use Haar cascade classifier to detect smiles in the face ROI
                smiles = smile_cascade.detectMultiScale(gray[face.top():face.bottom(), face.left():face.right()],
                                                         scaleFactor=1.8, minNeighbors=20, minSize=(30, 30))

                # Use Haar cascade classifier to detect eyes in the face ROI
                eyes = eyes_cascade.detectMultiScale(gray[face.top():face.bottom(), face.left():face.right()],
                                                       1.3, 5)

                # If both smiles and eyes are detected
                if len(smiles) > 0 and len(eyes) > 0:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, "TRUTH", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                # If no smiles are detected, consider it as a potential lie
                else:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                    cv2.putText(frame, "LIE", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)

        # Display the frame
        cv2.imshow('Lie Detection', frame)

        # Check if user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
