max_length = 1500
max_features = 20000

path = '/home/wzc/data/2019-3-19/'
pk_path = 'cnews_pk.pkl'
# model_path = 'cnews_model.model'

import pickle as pk
from keras.callbacks import ModelCheckpoint
from keras.models import *
from keras.layers import *
from keras.layers.convolutional import Convolution2D
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
'''get data[0]、label[1]、val_data[2]、val_label[3]、embedding_matrix[4] from cnews_pk.pkl'''
# pk_get = open(path + pk_path, 'rb')
# pk_store = pk.load(pk_get)
# data = pk_store[0]
# label = pk_store[1]
# val_data = pk_store[2]
# val_label = pk_store[3]
# embedding_matrix = pk_store[4]
# pk_get.close()
# print(data)
# print(len(data))
# print(label)
# print(len(label))
# print(embedding_matrix)
# print(len(embedding_matrix))


read = open(path + 'cnews.conclude', 'rb')
conclude = pk.load(read)
X_train = conclude[0]
X_test = conclude[1]
y_train = conclude[2]
y_test = conclude[3]
val_data = conclude[4]
val_label = conclude[5]
# embedding_weights = conclude[4]

print(y_train)

'''main：'''
inputs = Input(shape=(max_length,), dtype='int32')

# emb = Embedding(100000, 100, input_length=max_length, trainable=True)(inputs)
# emb = Embedding(max_features, max_length, weights=[embedding_weights], input_length=max_length, trainable=True)(inputs)
emb = Embedding(max_features, max_length, input_length=max_length, trainable=True)(inputs)
# emb = LSTM(100, return_sequences=True)(emb)

'''LSTM'''
# x = LSTM(200, activation='tanh', dropout=0.5, return_sequences=True)(emb)
# x = Flatten()(x)
# model_path = 'cnews_LSTMmodel.model'

'''LSTM'''
x = LSTM(200, activation='tanh', dropout=0.5, return_sequences=True)(emb)
x = Convolution1D(filters=64, kernel_size=3, strides=3, padding='valid', activation='relu')(x)
x = MaxPooling1D()(x)
x = Flatten()(x)
model_path = 'cnews_LSTMaddConvolution1DandMaxPooling1D_model.model'

'''RNN'''
# x = SimpleRNN(200, activation='tanh', dropout=0.5, return_sequences=False)(emb)
# x = Dense(200, activation='tanh')(x)
# # model_path = 'cnews_SimpleRNNmodel.model'
# model_path = 'cnews_SimpleRNN_addedDense_model.model'

'''CNN'''
# x = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(emb)
# x = Flatten()(x)
# model_path = 'cnews_Conv1Dmodel.model'

'''CNN'''
# x = Convolution1D(filters=64, kernel_size=3, strides=3, padding='valid', activation='relu')(emb)
# x = MaxPooling1D()(x)
# x = Convolution1D(filters=128, kernel_size=3, strides=3, padding='valid', activation='relu')(x)
# x = MaxPooling1D()(x)
# x = Dropout(0.5)(x)
# x = Flatten()(x)
# model_path = 'cnews_CNNadd_convolution1DandMaxPolling1Ddoubled_model.model'

'''output'''
output = Dense(10, activation='softmax')(x)


model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(path + model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(X_train, y_train, epochs=15, callbacks=[checkpoint], validation_data=[X_test, y_test])
prediction = np.argmax(model.predict(val_data), axis=1)
print(prediction.shape)
accuracy = accuracy_score(val_label, prediction)
f1 = metrics.f1_score(val_label, prediction, average='weighted')
print('acc=')
print(accuracy)
print('f1=')
print(f1)