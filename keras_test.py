path = '/home/wzc/data/2019-3-19/'
data_path = 'cnews_data.txt'
label_path = 'cnews_label.txt'
word_num_path = 'cnews_word_num.txt'
word2vec_model_path = 'word2vec_model'
max_length = 1500
max_features = 20000

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import *
from keras.layers import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim
from gensim.models import word2vec
# def train_test_split(data, label, test_size):


'''get data'''
read_data = open(path + data_path, 'r')
data = []
for item in read_data.readlines():
    items = item.strip().split('\n')
    data.append(items)
read_data.close()
# print(data)

'''get label'''
read_label = open(path + label_path, 'r')
label = []
for item in read_label.readlines():
    label.append(item.strip())
read_label.close()
# print(label)

'''get word_num'''
read_word_num = open(path + word_num_path, 'r')
word_num = []
for item in read_word_num.readlines():
    word_num.append(item.strip().split('\t')[0])
read_word_num.close()
# print(word_num)


'''data prepocessing'''
sentences = word2vec.Text8Corpus(path + data_path)
word2vec_model = word2vec.Word2Vec(sentences)
word2vec_model.save(path + word2vec_model_path)

train_data = []
with open(path + data_path) as f:
    train_data = f.read().splitlines()
print('this is train_data\'s length:')
print(len(train_data))
# print('this is train_data')
# print(train_data)
tokenizer = Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
# print('word_index:')
# print(word_index)

sequences = tokenizer.texts_to_sequences(train_data)

# data = pad_sequences(sequences, maxlen=max_features)
data = tokenizer.sequences_to_matrix(sequences, mode='binary')
print('data:')
print(data)



# tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
# tokenizer.fit_on_texts()


'''label prepocessing'''
# print(label)
label = to_categorical(label)
print('label:')
print(label)
print(label.shape)

'''Embedding'''
model = gensim.models.Word2Vec.load(path + word2vec_model_path)
word2idx = {"_PAD": 0}
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embedding_matrix[i + 1] = vocab_list[i][1]

'''main'''
# print(data.shape)

inputs = Input(shape=(max_features,))

# emb = Embedding(output_dim=512, input_dim=max_features, input_length=max_length)(inputs)
emb = Embedding(len(embedding_matrix), 100, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs)

x = Dense(64, activation='relu')(emb)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(data, label, epochs=2)



# inputs = Input(shape=(max_features,), dtype="int32")
# emb = Embedding(output_dim=data.shape[1], weights=[data], input_dim=data.shape[0],)(inputs)
#
# conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=64, kernel_size=3, padding="valid")(emb)))
#
# x = Dense(64, activation='relu')(conv)
# output = Dense(10, activation='softmax')(x)
#
# model = Model(inputs=inputs, outputs=output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(data, label, epochs=1)