import cv2
import dlib
from scipy.spatial import distance

def euclidean_distance(eye):
    A=distance.euclidean(eye[0],eye[3])
    B=distance.euclidean(eye[1],eye[5])
    C=distance.euclidean(eye[2],eye[4])

    dist=(B+C)/(2.0*A)
    return dist

cap=cv2.VideoCapture(0)
get_face=dlib.get_frontal_face_detector()
face_marks=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _,frame=cap.read()


    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = get_face(gray)


    for face in faces:

        face_segments=face_marks(gray,face)
        left_eye = []
        right_eye = []

        for i in range(36,42):
            x=face_segments.part(i).x
            y=face_segments.part(i).y
            next_point=i+1
            left_eye.append((x,y))
            if i==41:
                next_point=36
            x2=face_segments.part(next_point).x
            y2=face_segments.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(255,0,0),1)

        for i in range(42,48):
            x=face_segments.part(i).x
            y=face_segments.part(i).y
            next_point=i+1
            right_eye.append((x,y))
            if i==47:
                next_point=42
            x2=face_segments.part(next_point).x
            y2=face_segments.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(255,0,0),1)

        left_dist=euclidean_distance(left_eye)
        right_dist=euclidean_distance(right_eye)


        eye_dist=(left_dist+right_dist)/2.0
        eye_dist=round(eye_dist,2)

        if eye_dist<0.28:
            cv2.putText(frame,"Drowsy",(20,100),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),1)
            cv2.putText(frame,"Are you sleepy..?",(20,400),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),1)
        else:
            cv2.putText(frame,"Wide Awake",(20,100),cv2.FONT_HERSHEY_DUPLEX,3,(0,0,255),2)
    cv2.imshow("Are you Sleepy", frame)

    key=cv2.waitKey(27)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()







