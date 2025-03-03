import cv2
import mediapipe as mp

def generateFace(imageArray):
    #change to RGB
    # imageArray=cv2.cvtColor(imageArray,cv2.COLOR_BGR2RGB)

    #Now create mp instance
    mp_faceDetection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    #detect the faces
    with mp_faceDetection.FaceDetection() as faceDetection:
        results=faceDetection.process(imageArray)

    #process and output detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = imageArray.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Extract face
            face = imageArray[y:y + h_box, x:x + w_box]
            return face
    else:
        return None


def readCameraStream(camIndex):
    cap1 = cv2.VideoCapture(camIndex)
    videoOutputStatus = False
    # set video capture output size to 700 and 400
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 700)

    while True:
        readStatus, frame = cap1.read()
        print(readStatus)
        #extract face area from frame
        face=generateFace(frame)
        if face is not None:
            cv2.imshow('Output', face)
        else:
            cv2.imshow('Output', frame)  # Show original frame if no face is detected

        # cv2.imshow('Output',face) #this one outputs image
        # faceAreaGeneration(frame) #this one....well shit
        if cv2.waitKey(10)==ord('q'):
            cv2.imwrite('capturedFace.jpg',face)
            cv2.destroyWindow('Output')
            videoOutputStatus=True
            break

    if videoOutputStatus:
        print('Video capture stopped')



readCameraStream(1)