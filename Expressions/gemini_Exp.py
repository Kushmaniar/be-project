import google.generativeai as genai

import PIL.Image

def exp_recog(img):
    GOOGLE_API_KEY = "AIzaSyCFb7nxEauCLQ7FAW2odYHkD1oJNmCx264"

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro-vision')
    # img = PIL.Image.open('image.jpg')

    response = model.generate_content(['''
                                       Classify the emotion of the person in the image based on the following emotions - 
                                       1. Reply 'Positive' if the emotion is positive (eg. smiling, laughing, comforting)
                                       2. Reply 'Negative' if the emotion if negative (eg. frowning, grimacing, anger)
                                       3. Reply 'Neutral' if the emotion is neutral (eg. poker face, neutral stare)
                                       4. Reply 'Other' if the emotion is other than the above categories (eg. crying, confused)

                                       ## ONLY REPLY IN ONE WORD 
                                       ''', img])
    response = response.text

    return response

# img = PIL.Image.open('smile.jpg')
# print(exp_recog(img))