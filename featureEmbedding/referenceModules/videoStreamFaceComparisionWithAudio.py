import cv2
import mediapipe as mp
import faceEmbedding
import threading


def generateFace(imageArray):
    #change to RGB
    #imageArray=cv2.cvtColor(imageArray,cv2.COLOR_BGR2RGB)

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

    frameIndex=0


    while True:
        frameIndex+=1
        if frameIndex==30:
            #start a parallel thread
            # compare the two images
            face = generateFace(frame)
            frameIndex = 1
            threading.Thread(target=faceEmbedding.compare_faces, args=(face,refImage,)).start()


            # faceEmbedding.compare_faces(face,refImage) #array, image
            if faceEmbedding.similar==True:
                print("Face Recognized")

        readStatus, frame = cap1.read()
        # print(readStatus)
        #extract face area from frame

        cv2.imshow('VidStream',frame)


        if cv2.waitKey(10)==ord('q'):
            # cv2.imwrite('sample1.jpg',face)
            cv2.destroyWindow('Output')
            videoOutputStatus=True
            break

    if videoOutputStatus:
        print('Video capture stopped')


refImage='capturedFace.jpg'
readCameraStream(1)
