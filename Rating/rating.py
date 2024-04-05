
import joblib
import PIL
# from ..LDA.newlda import lda_full 
# from finalyearproject.LDA.newlda import lda_full
import sys
sys.path.append('../')

from LDA import newlda
from Expressions import gemini_Exp

model = joblib.load('linear_reg_model.joblib')

sample_text = '''
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
#parameter generation

#1. ayush's gemini params
#2. database params

#3. lda to keywords
def get_lda_param(text):
    keywords = newlda.lda_full(text)
    return keywords

#4. face expressions
def get_exp_param(img):
    expression_type = gemini_Exp.exp_recog(img)
    return expression_type

def pipeline(sample_text, img):
    keywords = get_lda_param(sample_text)
    expression_type = get_exp_param(img)

    return [keywords, expression_type]


img = PIL.Image.open('..\\Expressions\\smile.jpg')
print(pipeline(sample_text, img))
