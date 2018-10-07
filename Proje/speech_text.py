import speech_recognition as sr


def listen():

  r=sr.Recognizer()

  with sr.Microphone() as source:

      print('Say something')
      audio=r.listen(source)
      print('Done!')

  text=r.recognize_google(audio,language='tr-TR')

  print(text)






