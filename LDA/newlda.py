text = '''
Hey John, how are you doing today?
David? Is that you? Haven't seen you in a while. Where's Mary?
She's at the garden club meeting, John. Remember? We talked about it last week.
Last week? Did we? Feels like ages ago. Mind you, everything feels like ages ago these days.
I know it's tough, John. But that's why I'm here. Brought you some of your favorites – apples and that oatcake you like.
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
'''

import pandas as pd
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from .gemini_LDA import lda_to_keywords

def lemmatize_stemming(text):
    wnl = WordNetLemmatizer()
    lem_word = wnl.lemmatize(text)
    ps = PorterStemmer()
    lem_stemmed_word = ps.stem(lem_word)
    return lem_word

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result

import re
def remove_newlines(text):
    cleaned_text = re.sub(r'\n', '', text)
    return cleaned_text

def lda_full(ogtext):
    text = remove_newlines(ogtext)
    sents = sent_tokenize(text)

    # Preprocess the text data
    texts = [preprocess(text) for text in sents]

    combined_data = pd.DataFrame({'Text': texts})

    dictionary = gensim.corpora.Dictionary(combined_data['Text'])

    # print(dictionary)
    # count = 0
    # for k, v in dictionary.iteritems():
    #     print(k, v)
    #     count += 1
    #     if count > 10:
    #         break

    bow_corpus = [dictionary.doc2bow(doc) for doc in combined_data['Text']]

    bow_doc = bow_corpus[1]

    for i in range(len(bow_doc)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc[i][0],
                                                        dictionary[bow_doc[i][0]],
                                                        bow_doc[i][1]))
        
    lda_model = gensim.models.LdaModel(corpus=bow_corpus, num_topics=5, id2word = dictionary, chunksize=20000, passes=2)

    lda_string = ''''''
    for idx, topic in lda_model.print_topics(-1):
        lda_string = lda_string + f"Topic: {idx} \nWords: {topic} \n\n"
        # print("Topic: {} \nWords: {}".format(idx, topic))
        # print("\n")

    keywords = lda_to_keywords(ogtext, lda_string)
    return keywords

# response = lda_full(text)
# print('response', response)
