
import speech_recognition as sr
from multiprocessing import Process
import tensorflow as tf
import pyttsx3 # python based text-to-speech engine
import numpy as np
import urllib3        #maybe requests is another good choice
import serial
import time
import fire
import json
import os
import sys
import cv2

engine = pyttsx3.init()

def processVideo():

    def getCommonColors(url):

        req = urllib2.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr,-1)

    arduino = serial.Serial('COM5', 19200)
    time.sleep(2)
    print("Connection to arduino...")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        cv2.resizeWindow('img', 600,450)
        gray  = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (155,255,0), 1)


        for (x,y,w,h) in faces:
            w2 = int(round(w/2))
            h2 = int(round(h/2))
            cv2.circle(img,(x+(w2),y+(h2)),(h2),(255,255,0),1,cv2.LINE_AA)
            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            arr = {y:y+h, x:x+w}
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:

                    ew2 = int(round(ew/2))
                    eh2 = int(round(eh/2))
                    cv2.circle(roi_color,(ex+(ew2),ey+(eh2)),(eh2),(255,255,255),-1)

            #print (arr)

            #print ('X :' +str(x))
            #print ('Y :'+str(y))
            #print ('x+w :' +str(x+w))
            #print ('y+h :' +str(y+h))

            xx = int(x+(x+h))/2
            yy = int(y+(y+w))/2
            zz = int((h+w)/2)


            #print (xx)
            #print (yy)

            center = (xx,yy)
            global centered

            if(xx <= 320 and xx >= 280):
                centered = 1
            elif(yy <= 270 and yy >= 230):
                centered = 1
            else:
                centered = 0
            cc = bool(centered)

            data = "X{0:n}Y{1:n}Z{2:n}C{3:b}".format(xx, yy, zz, cc)
            print(data)
            arduino.write(data.encode())



        cv2.imshow('img',img)


        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

def processSpeech():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Initialized")
        audio = r.listen(source)
    # recognize speech using Sphinx
    flag=True
    while(flag==True):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            test = r.recognize_sphinx(audio)
            engine.say(test)
            print(test)
            engine.runAndWait()
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))

'''

Erase Comment if placed in base dir of GPT2 with model '355M' installed
 


def gpt2():

    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration = 1)
        print("calibrating")
        audio = r.listen(source)
        print("respond vocally")

    import model, sample, encoder

    def interact_model(
        model_name='355M',
        seed=None,
        nsamples=1,
        batch_size=1,
        length=30,
        temperature=40,
        top_k=40,
        models_dir='models',
    ):
        """
        Interactively run the model
        :model_name=117M : String, which model to use
        :seed=None : Integer seed for random number generators, fix seed to reproduce
         results
        :nsamples=1 : Number of samples to return total
        :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
        :length=None : Number of tokens in generated text, if None (default), is
         determined by model hyperparameters
        :temperature=1 : Float value controlling randomness in boltzmann
         distribution. Lower temperature results in less random completions. As the
         temperature approaches zero, the model will become deterministic and
         repetitive. Higher temperature results in more random completions.
        :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
         considered for each step (token), resulting in deterministic completions,
         while 40 means 40 words are considered at each step. 0 (default) is a
         special setting meaning no restrictions. 40 generally is a good value.
         :models_dir : path to parent folder containing model subfolders
         (i.e. contains the <model_name> folder)
        """
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k
            )

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)

            while True:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    print("listening")
                    audio = r.listen(source)
                    try:
                        raw_text = r.recognize_sphinx(audio)
                        raw_text = raw_text.lower()
                    except sr.UnknownValueError:
                        print("sorry, i didn't catch that")
                        raw_text = r.recognize_sphinx(audio)
                    except sr.RequestError as e:
                        print("sorry, i didn't catch that")
                        raw_text = r.recognize_sphinx(audio)
                while not raw_text:
                    engine.say("sorry, i didn't catch that")
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print(text)
                        engine.say(text)

if __name__ == '__main__':
  p1 = Process(target=processVideo)
  p1.start()
  p2 = Process(target=gpt2)
  p2.start()
  p1.join()
  p2.join()
'''
