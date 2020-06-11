from flask import Flask,render_template,Response
import tensorflow as tf
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

'''
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
'''

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =tf.keras.models.load_model('vgg_model_saved.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
'''


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_frame(self):
        while True:
        # Grab a single frame of video
            ret, frame = self.cap.read()
            frame=cv2.flip(frame,1)
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

            _,jpeg=cv2.imencode('.jpeg',frame)


            return jpeg.tobytes()
            '''
            cv2.imshow('Emotion Detector',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''

    def __del__(self):            
        self.cap.release()
        cv2.destroyAllWindows()

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(Camera):
    while True:
        frame=Camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +frame+ b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__ == '__main__':
	app.run(debug=True)
				
