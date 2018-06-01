import sys
import io
import argparse

import numpy as np
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import load_model






def data_matrix_to_conllu(x, x_strides, y, vocab, f=sys.stdout):

    #make a reverse vocabulary
    rev_vocab = {v:k for k,v in vocab.items()}
    out = ''
    cntr = 1

    y = y.argmax(-1)

    current_token = ''
    current_sent = []
    sents = []

    def write_conllu(sent, f):
        
        for i, t in enumerate(sent):
            line = [str(i+1), t.lstrip(' ')]
            line.extend(['_']*7)
            
            '''
            if i > 0:
                line[6]='1'
            else:
                line[6]='0'
            '''

            if len(sent)-2 > i:
                if sent[i+1] == ' ':
                    line.append('SpaceAfter=False')
                else:
                    line.append('_')
            else:
                line.append('_')
            f.write('\t'.join(line)+'\n')
        f.write('\n')

    for a, line in enumerate(x):
        for b, char in enumerate(line):

            try:
                current_token += x_strides[a][b]
            except: pass

            if y[a][b] == 1.0:
                if len(current_token.lstrip(' ')) > 0:
                    current_sent.append(current_token)
                current_token = ''

            if b > -1 and y[a][b] == 2.0:

                if len(current_token.lstrip(' ')) > 0:
                    current_sent.append(current_token)
                current_token = ''

                write_conllu(current_sent,f)
                current_sent = []
                cntr = 1
    else:
        if current_sent:
            write_conllu(current_sent,f)


def make_data_matrix(x, width = 150, vocab=None):

    x_strides = []
    y_1_str = []
    y_2_str = []

    for idx in range(0,len(x), width):
        x_strides.append(x[idx:idx+width])

    #Build vocab
    if vocab == None:
        #
        vocab = {'<mask>': 0.0}

    #Update vocab
    '''
    for k in x:
        if k not in vocab.keys():
            #
            vocab[k] = len(vocab)
    '''

    #Let's build 
    xnp = np.zeros((len(x_strides), width))

    ##
    for a, line in enumerate(x_strides):
        for b, char in enumerate(line):
            #
            try:
                xnp[a][b] = vocab[x_strides[a][b]]
            except:
                pass


    return xnp, x_strides


#Moved here to avoid messy tensorflow imports in tokenizer_server.py
def tokenize_text(txt,model,vocab, sentence_mode=False):
    buff=io.StringIO()
    
    x = []

    curr_block = []
    for line in txt.split('\n'):
        if line.startswith('###'):

            if len(curr_block) > 0:

                xx, x_strides = make_data_matrix(list(' '.join(curr_block)), vocab=vocab)
                pred = model.predict(xx)
                data_matrix_to_conllu(xx, x_strides, pred, vocab, f=buff)

                curr_block = []
                buff.write(line)
        else:

            if sentence_mode:

                xx, x_strides = make_data_matrix(list(line.strip()), vocab=vocab)

                pred = model.predict(xx)
                data_matrix_to_conllu(xx, x_strides, pred, vocab, f=buff)
                curr_block = []

            else:
                curr_block.append(line.strip())





    xx, x_strides = make_data_matrix(list(' '.join(curr_block)), vocab=vocab)

    print (x_strides)

    pred = model.predict(xx)
    data_matrix_to_conllu(xx, x_strides, pred, vocab, f=buff)

    return buff.getvalue()

if __name__ == '__main__':

    import pickle

    model = load_model('model')
    vocab = pickle.load(open('vocab.pickle','rb'))


    test_txt = '''Ruotsin paras "suomalainen" yleisurheilija Daniel Ståhl pettyi pahasti Rooman Timanttiliigan kilpailussa torstai-iltana. Ståhl jäi itselleen vaatimattomalla tuloksella 64,84 neljänneksi.

Edelle kiilasivat Jamaikan Fedrick Dacres (68,51), Liettuan Andrius Gudzius (68,17) ja Iranin Ehsan Hadadi (65,93).

Viime vuonna MM-hopeaa voittaneen Ståhlin ennätys on 71,29.

Tukholmassa asuva Ståhl puhuu sujuvaa suomea.

- Se on se suomalainen sisu ruotsalaisessa miehessä. Kun menee huonosti, kiroilen suomeksi, Ståhl kertoi viime vuonna Iltalehden haastattelussa.

Ståhlin äiti Taina Laakso on suomalainen.!?'''

    print (tokenize_text(test_txt, model, vocab))





