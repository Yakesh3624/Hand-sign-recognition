import cv2
import numpy
from keras.models import model_from_json
from keras.preprocessing import image
cam = cv2.VideoCapture(0)

model = open(r"D:\Pantech ai\basic pgm in keras & tensorflow\Hand sign recognition\model.json")
model=model.read()
model = model_from_json(model)

model.load_weights(r"D:\Pantech ai\basic pgm in keras & tensorflow\Hand sign recognition\model.h5")
while True:

    img = cam.read()[1]
    i = cv2.resize(img,(400,400))
    i = image.img_to_array(i)
    i = numpy.expand_dims(i,axis=0)

    #print(i.shape)
    classes = ['0','1','2','3','4','5','6','7','8','9']

    prediction = model.predict(i)[0]
    text=''
    for i in range(len(prediction)):
        if prediction[i]==1:
            text=classes[i]
    cv2.putText(img,text,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("img",img)
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break
