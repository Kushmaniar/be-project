# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

from LDA import newlda
from Expressions import gemini_Exp

#Imprts for gemini
import os
import json
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt

import google.generativeai as genai
GOOGLE_API_KEY = "AIzaSyDp3xiMDHVsYO7BKZh13-BvhnGK9aS4sFs"



    
# Imports for transcription
import whisper
import pyaudio 
import wave
import subprocess
import json
from vosk import Model, KaldiRecognizer

#Imports for regression
import joblib

# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import keras

from cvzone.FaceMeshModule import FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
# Detect faces 

from layers import L1Dist

sample_conv ="""
    Hey John, how are you doing today?
    David? Is that you? Haven't seen you in a while. Where's Mary?
    She's at the garden club meeting, John. Remember? We talked about it last week.
    Last week? Did we? Feels like ages ago. Mind you, everything feels like ages ago these days.
    I know it's tough, John. But that's why I'm here. Brought you some of your favorites - apples and that oatcake you like.
    Oatcake! Now you're talking.Mary always hides them, says I eat them all at once.
    Well, maybe I can convince her to let you have a little more freedom with the snacks today.  So, what have you been up to, champ?
    This… this thing. Can't remember what it's called. Used to play chess all the time with… with… 
    Your dad? You and your dad used to have epic chess battles. Remember how you'd always try to trick him with that sneaky queen move?
    Dad? Right! Dad. Used to love playing with him. He always beat me, though. Smart man, your dad.
    He was a sharp one, that's for sure. Hey, listen, how about we set up a game? Maybe I can get schooled by the master himself?
    Master? Hardly. But… a game sounds good. Been a while since I've played. Remember the rules, though?
    Of course, John. How about we play for the oatcake? Winner takes all?
    Winner takes all? You're on! Don't underestimate the old dog, David.
    Wait… where's the knight?
    Right here, John. Next to your king. See? Remember, the knight moves in an L-shape.
    Ah, the knight. Right. Used to love using the knight for surprise attacks. Like a sneaky little soldier.
    My, look at the time! Mary will be home soon.
    Don't worry about Mary, John. We still have a few moves left in us, don't we?  Besides, who knows, maybe you can use that sneaky knight to win the oatcake after all.
    """

def gem():

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    system_prompt = f"""
    %INSTRUCTIONS
    You are a medically trained AI conversationalist who is extremely profecient in understanding human conversations.
    I want you to semantically understand and comprehend a conversation between an alzheimer's patient and a visitor.
    
    % YOUR TASK 
    You need to extract the following parameters from the conversation and provide values for each.
    # Strictly keep the values of the parameters within the range mentioned in brackets next to them.

    %INPUT
    Textual Conversation: {sample_conv}
    
    %OUTPUT FORMAT

    ##Strict CSV output only, no comments and text expected!
    ##Every new line in the csv output has to be strictly in numerical format ONLY and shouldn’t contain any textual information!
    Parameters - 
    Emotional Content - happiness(3), comfort(1), neutral(0), confusion(-1), frustration(-2), fear(-3)
    Recollection Accuracy - 0-5 (0 - no recollection at all, 5 - complete recollection)
    Visito's Empathy and Understanding - 0-5 (0 - no empathy at all, 5 - complete empathy)
    Patient's Interaction Level - 0-5 (0 - no interaction, 5 - heavy interaction) 


    %SAMPLE OUTPUT:
    "
    Emotional Content,Recollection Accuracy,Visitor Empathy,Patient Interaction Level
    2,5,3,3
    "


"""
    chat = model.start_chat(history=[])
    response = chat.send_message(system_prompt)
    response = response.text
    with open('res.csv', 'w') as my_file:
        my_file.write(response)
    print("Gem done")

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

def transcribe():

    CHUNK=1024
    FORMAT=pyaudio.paInt16
    CHANNELS=1
    RATE=16000
    RECORD_SECONDS=5


    WAVE_OUTPUT_FILENAME="chunk.wav"

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=2
    )

    print("* recording")
    frames=[]
    for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
        data=stream.read(CHUNK)
        frames.append(data)


    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    model = whisper.load_model("base")
    result = model.transcribe("chunk.wav", fp16=False, language="en")
    return result['text']

def regression(name, verified):
    lin_reg_model = joblib.load('linear_reg_model.joblib')
    preddata = pd.read_csv('res.csv')
    with open('database.json', 'r') as f:
        db = json.load(f)
    preddata['Visitor Frequency'] = db[name]['frequency']
    preddata['Recognized Visitor'] = verified 
    preddata['Type of Relation'] = db[name]['relationship']
    # lda_score = newlda.lda_full(sample_conv)
    preddata['LDA Topic Relevance'] = 3
    input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
    expression_type = gemini_Exp.exp_recog()
    preddata['Facial Expressions']=expression_type
    pred = lin_reg_model.predict(preddata)
    print("trust rating",pred)
    



def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    name = ""
    for person in os.listdir(os.path.join('application_data', 'verification_images')):
        for image in os.listdir(os.path.join('application_data', 'verification_images', person)):
            input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = preprocess(os.path.join('application_data', 'verification_images', person, image))
            
            # Make Predictions 
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images', person))) 
        verified = verification > verification_threshold
        if verified:
            name = person
            break
    if name == "":
        return False, name
    else:
        return True, name

# Reload model
# custom_objects = {"L1Dist": L1Dist}

# with keras.saving.custom_object_scope(custom_objects):
#     siam_model = keras.models.load_model("siamesemodelv2.h5")

siamese_model = keras.models.load_model('siamesemodelv3.keras', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # cropframe = frame[235:235+250,515:515+250, :]

    bfaces = face_cascade.detectMultiScale(frame, 1.2, 10, minSize=(64,64))
    for (x, y, w, h) in bfaces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h),  
                    (0, 0, 255), 2) 
        facess = frame[y+50:y + 300, x+50:x + 300] 
        # cv2.imshow("face",facess) 
    frame, faces = detector.findFaceMesh(frame, draw=False)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w,_ = detector.findDistance(pointLeft, pointRight)
        W=6.3
        f=1000

        #Finding depth
        d=(W*f)/w
    
    
    # Verification trigger
        agg_verification = 0
        proxim = False
        # if key in [27, ord('a'), ord('A')]:
        #     for i in range(5):
        #         imgname = os.path.join('application_data', 'verification_images', '{}_{}.jpg'.format(name, uuid.uuid1()))
        #         cv2.imwrite(imgname, facess)

        if d<45:
            proxim = True

            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), facess)
            # Run verification
            verified, name = verify(siamese_model, 0.5, 0.5)
            verified =1 if verified else 0
            # cvzone.putTextRext(frame, )   
        if proxim==True and verified:
            print("Verified", name)
            conversation = transcribe()
            print("Conversation:", conversation)
            gem()
            regression(name,verified)
        elif proxim == True and not verified:
            print("Unverified")
        # gem()
    cv2.imshow("Full frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()