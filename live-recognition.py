import cv2
import mtcnn
from mtcnn.mtcnn import MTCNN
print(mtcnn.__version__)

# creating detector 
detector = MTCNN()
print("Detector activated!!!!")

# initialize the capture object
cap = cv2.VideoCapture(0)

# get the video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a video writer object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output1.mp4', fourcc, 20.0, (width, height))

# loop through frames and capture video
while cap.isOpened():
    # read a frame from the camera
    ret, frame = cap.read()

    if ret:
        # detect faces in the frame
        faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        f = frame

        # draw bounding boxes around the faces detected
        for result in faces:
            x, y, w, h = result['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # write the processed frame to the output video
        writer.write(frame)

        # display the processed frame
        cv2.imshow('Face Detection', frame)

    # break if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
writer.release()
cv2.destroyAllWindows()
