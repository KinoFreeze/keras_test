
path = '/home/wzc/data/2019-3-19/'
pk_path = 'cnews_pk.pkl'
# model_path = 'cnews_LSTMmodel.model'
# model_path = 'cnews_SimpleRNNmodel.model'
# model_path = 'cnews_SimpleRNN_addedDense_model.model'
# model_path = 'cnews_Conv1Dmodel.model'
# model_path = 'cnews_CNNadd_convolution1DandMaxPolling1Ddoubled_model.model'
model_path = 'cnews_LSTMaddConvolution1DandMaxPooling1D_model.model'
import pickle as pk
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.models import load_model
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# class Metrics(Callback):
#   def on_train_begin(self, logs={}):
#     self.val_f1s = []
#     self.val_recalls = []
#     self.val_precisions = []
#
#   def on_epoch_end(self, epoch, logs={}):
#     val_predict=(np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#     val_targ = self.model.validation_data[1]
#     _val_f1 = f1_score(val_targ, val_predict)
#     _val_recall = recall_score(val_targ, val_predict)
#     _val_precision = precision_score(val_targ, val_predict)
#     self.val_f1s.append(_val_f1)
#     self.val_recalls.append(_val_recall)
#     self.val_precisions.append(_val_precision)
#     print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))
#     return
#
# metrics = Metrics()




read = open(path + 'cnews.conclude', 'rb')
conclude = pk.load(read)
X_train = conclude[0]
X_test = conclude[1]
y_train = conclude[2]
y_test = conclude[3]
val_data = conclude[4]
val_label = conclude[5]

model = load_model(path + model_path)
prediction = np.argmax(model.predict(val_data), axis=1)
print(prediction.shape)

accuracy = accuracy_score(val_label, prediction)
print('acc=')
print(accuracy)

f1 = metrics.f1_score(val_label, prediction, average='weighted')
print('f1=')
print(f1)

recall = metrics.recall_score(val_label, prediction, average='weighted')
print('recall=')
print(recall)

precision = metrics.precision_score(val_label, prediction, average='weighted')
print('precision=')
print(precision)