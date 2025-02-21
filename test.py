import cv2
import numpy as np
from keras.models import load_model # type: ignore

model = load_model('CNN_30E.keras')
cam = cv2.VideoCapture(0)

faceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

etiquetas_emociones = { 0:'Enojado', 1:'Disgustado', 2:'Asustado', 3:'Feliz', 4:'Neutral', 5:'Triste', 6:'Sorprendido'}

while True:
    ret, frame = cam.read()
    f_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetection.detectMultiScale(f_gris, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = f_gris[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img,(48,48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        # print(label)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w,y), (50,50,255), -1)
        cv2.putText(frame, 
                    etiquetas_emociones[label],
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2)
        
    cv2.imshow("Reconocimiento de emociones", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()