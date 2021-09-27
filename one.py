#-*- coding: utf-8 -*-
import os
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

#
os.environ['PYTHONHASHSEED'] = '0'
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)
import keras
from keras import backend as K
from stats_graph import stats_graph
from keras_layer_normalization import LayerNormalization
from keras.layers import Dense,Concatenate,Softmax,Conv1D,add,Masking,\
    GlobalAveragePooling1D,GlobalMaxPooling1D,Permute,Embedding,Add,Input,Flatten,Bidirectional,\
    Lambda,LSTM,Dense,Activation,AveragePooling1D,MaxPooling1D,multiply,concatenate,Dot,Dropout,BatchNormalization
from keras.models import Model
import data_helper
from keras.layers import Layer
from keras.utils.vis_utils import plot_model
from keras.callbacks import *

from tensorflow.python.ops.nn import softmax
from keras.utils.generic_utils import get_custom_objects
# from Trans import Transformer
import numpy as np
import random as rn
seed = 123
np.random.seed(seed)
rn.seed(seed)
tf.random.set_random_seed(seed)

input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
batch_size = 256
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)

def f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    recall = c1 / c3

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    return precision

def recall(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def loss(y_true, y_pred):
    margin = 0.7
    theta = lambda t: (K.sign(t) + 1.) / 2.

    return - (1 - theta(y_true - margin) * theta(y_pred - margin)
                - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
            ) * K.mean(y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))



margin = 0.75
theta = lambda t: (K.sign(t)+1.)/2.
nb_classes = 10



def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    in1_aligned = Add()([input_1,in1_aligned])
    in2_aligned = Add()([input_2,in2_aligned])
    return in1_aligned, in2_aligned



def base_network(input_shape):
    input = Input(shape=input_shape)
    em = embedding_layer(input)
    p1 = Bidirectional(LSTM(300, return_sequences=True, dropout=0.52), merge_mode='sum')(em)  # BQ 0.55 LCQMC 0.25
    p = concatenate([em, p1])
    model = Model(input, [p, em])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def siamese_model():
    input_shape = (input_dim,)


    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')
    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')
    input_q3 = Input(shape=input_shape, dtype='int32', name='sequence12')
    input_q4 = Input(shape=input_shape, dtype='int32', name='sequence22')



    q1_char, q1_em_char = base_network(input_shape)(input_q1)#600 300
    q2_char, q2_em_char = base_network(input_shape)(input_q2)#600 300
    q1_word, q1_em_word = base_network(input_shape)(input_q3)#600 300
    q2_word, q2_em_word = base_network(input_shape)(input_q4)#600 300


    q1_align_char,q2_align_char = align(q1_char,q2_char)
    q1_align_char = concatenate([q1_char,multiply([q1_char,q1_align_char])])
    q2_align_char = concatenate([q1_char,multiply([q2_char,q2_align_char])])
    q1_align_char,q2_align_char = align(q1_align_char,q2_align_char)


    q1_align_word,q2_align_word = align(q1_word,q2_word)
    q1_align_word = concatenate([q1_word,multiply([q1_word, q1_align_word])])
    q2_align_word = concatenate([q2_word,multiply([q2_word, q2_align_word])])

    q1_align_char,q1_align_word = align(q1_align_char,q1_align_word)
    q2_align_char,q2_align_word = align(q2_align_char,q2_align_word)

    q1_align = concatenate([q1_char,q1_word,q1_align_char,q1_align_word])
    q2_align = concatenate([q2_char,q2_word,q2_align_char,q2_align_word])

    q1_align = concatenate([GlobalAveragePooling1D()(q1_align),GlobalMaxPooling1D()(q1_align)])
    q2_align = concatenate([GlobalAveragePooling1D()(q2_align),GlobalMaxPooling1D()(q2_align)])


    #字词时序信息
    q1_char = concatenate([q1_char,q1_em_char])
    q1_char_ls = Bidirectional(LSTM(300, return_sequences=True, dropout=0.52), merge_mode='sum',name = 'q1_char_BIL')(q1_char)
    q1_char_ls_rc = concatenate([q1_char_ls,q1_em_char])


    q1_word = concatenate([q1_word, q1_em_word])
    q1_word_ls = Bidirectional(LSTM(300, return_sequences=True, dropout=0.52), merge_mode='sum',name = 'q1_word_BIL')(q1_word)
    q1_word_ls_rc = concatenate([q1_word_ls,q1_em_word])

    q1_char_ls_rc_align ,q1_word_ls_rc_align = align(q1_char_ls_rc,q1_word_ls_rc)





    q2_char = concatenate([q2_char, q2_em_char])
    q2_char_ls = Bidirectional(LSTM(300, return_sequences=True, dropout=0.52), merge_mode='sum',name = 'q2_char_BIL')(q2_char)
    q2_char_ls_rc = concatenate([q2_char_ls,q2_em_char])




    q2_word = concatenate([q2_word, q2_em_word])
    q2_word_ls = Bidirectional(LSTM(300, return_sequences=True, dropout=0.52), merge_mode='sum',name = 'q2_word_BIL')(q2_word)
    q2_word_ls_rc = concatenate([q2_word_ls,q2_em_word])


    q2_char_ls_rc_align,q2_word_ls_rc_align = align(q2_char_ls_rc,q2_word_ls_rc)

    q1_char_word = concatenate([q1_char,q1_word,q1_char_ls_rc_align,q1_word_ls_rc_align])
    q1_char_word = concatenate([GlobalAveragePooling1D()(q1_char_word), GlobalMaxPooling1D()(q1_char_word)])

    q2_char_word = concatenate([q2_char,q2_word,q2_char_ls_rc_align,q2_word_ls_rc_align])
    q2_char_word = concatenate([GlobalAveragePooling1D()(q2_char_word), GlobalMaxPooling1D()(q2_char_word)])

    q1 = concatenate([q1_char_word,q1_align])
    q2 = concatenate([q2_char_word,q2_align])

    q_all = multiply([q1,q2])

    q1_all = Lambda(lambda x:K.abs(x[0]-x[1]))([q1,q_all])
    q2_all = Lambda(lambda x:K.abs(x[0]-x[1]))([q2,q_all])
    similarity = Lambda(lambda x:K.abs(x[0]-x[1]))([q1_all,q2_all])



    similarity = Dropout(0.5)(similarity)
    similarity = LayerNormalization()(similarity)
    similarity = Dense(600,activation='relu')(similarity)
    similarity = LayerNormalization()(similarity)

    similarity = Dense(50,activation='relu')(similarity)
    similarity = LayerNormalization()(similarity)
    similarity = Dense(1)(similarity)
    pred = Activation('sigmoid')(similarity)





    model = Model([input_q1, input_q2, input_q3, input_q4], [pred])

    model.compile(loss=loss,optimizer='adam', metrics=['accuracy', precision, recall, f1_score])
    model.summary()
    return model


def train():
    
    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    print(type(train_q1))
    train_q2 = data['train_q2']
    train_q3 = data['train_q3']
    train_q4 = data['train_q4']
    train_y = data['train_label']



    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_q3 = data['dev_q3']
    dev_q4 = data['dev_q4']
    dev_y = data['dev_label']
    
    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_q3 = data['test_q3']
    test_q4 = data['test_q4']
    test_y = data['test_label']
    
    model = siamese_model()


    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, tensorboard,earlystopping,reduce_lr]
    print("*************")

    history = model.fit([train_q1,train_q2,train_q3,train_q4], train_y,
              batch_size=batch_size,
              epochs=200,
              validation_data=([dev_q1, dev_q2,dev_q3,dev_q4], dev_y),
              callbacks=callbackslist)



    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2,test_q3,test_q4],test_y,verbose=1,batch_size=256)
    print("Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score))

if __name__ == '__main__':
    train()