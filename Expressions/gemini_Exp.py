import google.generativeai as genai

import PIL.Image
import os

def exp_recog():
    img = PIL.Image.open(os.path.join('application_data','input_image','input_image.jpg'))
    GOOGLE_API_KEY = "AIzaSyCFb7nxEauCLQ7FAW2odYHkD1oJNmCx264"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro-vision')
    # img = PIL.Image.open('image.jpg')

    response = model.generate_content(['''
                                       Classify the emotion of the person in the image based on the following emotions - 
                                       'positive' = 4 (smiling, laughing, comforting), 
                                       'neutral' = 3 (poker face, neutral stare),
                                       'negative' = 2 (frowning, grimacing, anger), or 
                                       'other' = 1 (crying, confused).

                                       ## ONLY REPLY IN ONE NUMBER RANGING FROM 1-4 based on the emotion of the person
                                       ''', img])
    res = response.text

    return res

# img = PIL.Image.open('smile.jpg')
# print(exp_recog(img))