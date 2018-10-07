#coding-utf-8
from tkinter import Tk, Frame, Label, Button, LEFT, TOP, StringVar, Text
from tkinter import *
import cv2
import numpy as np
from PIL.ImageDraw2 import Font
from tkinter.font import Font
import util as ut
import svm_train as st
import speech_recognition as sr
# from gtts import gTTS
# import pyglet
# import time,os
# from pygame import mixer
# import time
# from colorama import Fore, Back, Style

model = st.trainSVM()

root = Tk()
root.title("Ses Ver")
frame1=Frame(root)
frame1.pack()
text61 = Text(root, height=50, width=30)
myFont = Font(family="Lucida Grande",size=16)


text61.configure(font=myFont,bg='#d8bfd8')
j=1


def start_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(4, 1024)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = " "
    temp = 0
    previouslabel = None
    previousText = " "
    label = None
    flag = True
    words=""

    def rescale_frame(frame, percent):
        scale_percent = 100
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    while (flag):

        _, img = cap.read()
        img = rescale_frame(img, percent=94)
        cv2.rectangle(img, (620, 100), (1000, 500), (255, 0, 0),
                      3)  # bounding box which captures ASL sign to be detected by the system
        img1 = img[100:500, 620:1000]
        img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
        blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)
        skin_ycrcb_min = np.array((0, 138, 67))
        skin_ycrcb_max = np.array((255, 173, 133))
        mask = cv2.inRange(blur, skin_ycrcb_min,
                           skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
        _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 2)
        cnt = ut.getMaxContour(contours, 4000)  # using contours to capture the skin filtered image of the hand
        # cnt!= None


        if cnt is not None:
            gesture, label = ut.getGestureImg(cnt, img1, mask,model)  # passing the trained model for prediction and fetching the result
            if (label != None):
                if (temp == 0):
                    previouslabel = label

            if previouslabel == label:
                previouslabel = label
                temp += 1
            else:
                temp = 0
            if (temp == 30):
               
                if (label == 'Q'):
                    label = " "
                text = text + label
                if (label == 'W'):
                    words = list(text)
                    print(words)
                    words.pop()
                    words.pop()

                    text = "".join(words)
                    # text=previousText
                print(text)

            cv2.imshow('PredictedGesture', gesture)  # showing the best match or prediction
            cv2.putText(img, label, (50, 150), font, 8, (0, 125, 155),
                        2)  # displaying the predicted letter on the main screen
            cv2.putText(img, text, (50, 450), font, 3, (0, 0, 255), 2)
        cv2.imshow('Frame', img)
        cv2.imshow('Mask', mask)
        k = 0xFF & cv2.waitKey(10)
        if k == 27:
            flag = False

            cv2.imshow('Frame', img)
            cap.release()

            words = list(text)
            print(words)
            words = "".join(words)
            text_inil ="[Kullanıcı 1] :"

            deneme =text_inil + words + "\n "
            text61.insert(INSERT, deneme.upper())
            print(deneme)

            print('aaaaaaaaaa')


def speech_text():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something')
        audio = r.listen(source)
        print('Done!')

    text_speech = r.recognize_google(audio, language='tr-TR')
    text_ini="[Kullanıcı 2] :"

    text_speech=text_ini + text_speech + "\n "
    text61.insert(INSERT, text_speech.upper())


start= Button(frame1, text="SIGN",width = 20,bg='red',command=start_webcam)
listen = Button(frame1,text="LISTEN",width = 20,bg='red',command=speech_text)
start.pack(side=LEFT)
listen.pack(side=RIGHT)
text61.pack(side=TOP)


root.mainloop()
