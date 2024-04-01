import cv2

# Load the Haar cascade classifiers for the face, smile, and eyes
face_cascade = cv2.CascadeClassifier("opencvlib/haarcascade_frontalface_default.xml")
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

        # Use Haar cascade classifier to detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi_gray = gray[y:y+h, x:x+w]

            # Use Haar cascade classifier to detect smiles in the face ROI
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(30, 30))

            # Use Haar cascade classifier to detect eyes in the face ROI
            eyes = eyes_cascade.detectMultiScale(roi_gray, 1.3, 5)

            # If both smiles and eyes are detected
            if len(smiles) > 0 and len(eyes) > 0:
                # Check the relationship between the positions of eyes and smile
                for (ex, ey, ew, eh) in eyes:
                    for (sx, sy, sw, sh) in smiles:
                        # Calculate parameters for more accurate detection
                        smile_area = sw * sh
                        eye_area = ew * eh
                        smile_intensity = smile_area / (w * h)
                        smile_eye_ratio = smile_area / eye_area

                        # Define thresholds for smile intensity and smile-eye ratio
                        intensity_threshold = 0.1
                        ratio_threshold = 0.5

                        # If the smile intensity and smile-eye ratio exceed thresholds, consider it as a genuine smile
                        if smile_intensity > intensity_threshold and smile_eye_ratio > ratio_threshold:
                            # Draw a rectangle around the face
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, "TRUTH", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            break

            # If no smiles are detected or if the smile intensity or ratio is below thresholds, consider it as a potential lie
            else:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "LIE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Lie Detection', frame)

        # Check if user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
