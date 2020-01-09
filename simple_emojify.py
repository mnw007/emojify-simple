import numpy as np
from emo_utils import get_emoji
from Emojify import *

is_first_time = True 
W,b = 0,0

def generate_emoji(sentence):
    global is_first_time,W,b
    if is_first_time:
        print('Press "q" to quit')
        W,b = model_new(X_train,Y_train,word_to_vec_map)
        is_first_time = False
    X = np.array([sentence])
    get_emoji(X,W,b,word_to_vec_map)
    
a = True
while a:
    inp = str(input('Enter sentence : '))
    if inp == 'q':
        a = False
        break
    generate_emoji(inp)

