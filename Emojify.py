import numpy as np
from emo_utils import *

X_train, Y_train = read_csv('train_emoji.csv')
X_test, Y_test = read_csv('tesss.csv')

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

def sentence_to_avg(sentence, word_to_vec_map):
    words = sentence.lower().split()

    avg = np.zeros((50,))
    
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    
    return avg

def model_new(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    
    np.random.seed(1)

    m = Y.shape[0]                          
    n_y = 5                                 
    n_h = 50                            
    
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    
    for t in range(num_iterations):                    
        for i in range(m):                             
            
            avg = sentence_to_avg(X[i], word_to_vec_map)

            z = np.dot(W, avg) + b
            a = softmax(z)

            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            W = W - learning_rate * dW
            b = b - learning_rate * db

    return W, b


