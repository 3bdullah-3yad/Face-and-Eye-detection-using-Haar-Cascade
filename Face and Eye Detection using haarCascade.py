import cv2

pretrained_model = r"D:\Projects\Computer Vision\Face Detection using HaarCascade\haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

input_path = r"D:\Projects\Computer Vision\Face Detection using HaarCascade\Ayad.mp4"
vid = cv2.VideoCapture(input_path)

w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', "V", "I", "D") # or *"XVID" / *"X264"
output_path = r"D:\Projects\Computer Vision\Face Detection using HaarCascade\Detected.mp4"
out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))


while vid.isOpened():

    done, frame = vid.read()
    if done==True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(gray_frame, 1.6, 12)
        for (x1, y1, w1, h1) in faces:
            cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
            
            rofInFrame = frame[y1:y1+h1, x1:x1+w1]
            rofInGrayFrame = gray_frame[y1:y1+h1, x1:x1+w1]
            
            eyes = eye_detector.detectMultiScale(rofInGrayFrame, 2.2, 8)
            for (x2, y2, w2, h2) in eyes:
                cv2.rectangle(rofInFrame, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)
        
        cv2.imshow("Ayad", cv2.resize(frame, (500, 900)))
        
        out.write(frame)
        
        if cv2.waitKey(1) == ord("a"):
            break
    else: break
cv2.destroyAllWindows()
vid.release()
out.release()