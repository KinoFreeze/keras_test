import pickle as pk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import gensim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

path = '/home/wzc/data/2019-3-19/'
read_path = 'cnews.train.txt'
read_val_path = 'cnews.val.txt'
data_path = 'cnews_data.txt'
pk_path = 'cnews_pk.pkl'
word2vec_model_path = 'word2vec_model'
model_path = 'cnews_model'
max_length = 1500
max_features = 20000


if __name__ == '__main__':

    label = []
    data = []

    read = open(path + read_path, 'r')
    for item in read.readlines():
        label.append(item[:2])
        data.append(item[3:])

    '''label2id的字典的生成'''
    label2id = {}
    count = 0
    for item in label:
        if item not in label2id:
            label2id[item] = count
            count += 1

    '''data label标签的存储'''
    labelpk = []
    for item in label:
        labelpk.append(label2id.get(item))

    '''data数据文本的存储'''
    data_list = []
    for item in data:
        content = ""
        for temp in item:
            content += temp + " "
        data_list.append(content)
    # del data_list[len(data_list) - 1]

    datapk = []
    for item in data:
        # print(item)
        content = ""
        for i in item:
            content += i + " "
        datapk.append(content)
        # datapk.append(item)

    X_train, X_test, y_train, y_test = train_test_split(datapk, labelpk, shuffle=True, test_size=0.2)
    print(y_train)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    '''val'''
    val_label = []
    val_data = []
    read = open(path + read_val_path, 'r')
    for item in read.readlines():
        val_label.append(item[:2])
        val_data.append(item[3:])

    val_data_pk = []
    for item in val_data:
        val_data_pk.append(item)

    val_data_list = []
    for item in val_data:
        content = ""
        for temp in item:
            content += temp + " "
        val_data_list.append(content)
    # del val_data_list[len(val_data_list) - 1]

    val_label_pk = []
    for item in val_label:
        val_label_pk.append(label2id.get(item))

    write_data = open(path + data_path, 'w', encoding='utf-8')
    write_data.close()
    write_data = open(path + data_path, 'a', encoding='utf-8')
    write_data.write(''.join(data_list))
    write_data.write('\n')
    write_data.write(''.join(val_data_list))
    write_data.close()

    '''embedding'''
    # sentences = word2vec.Text8Corpus(path + data_path)
    # word2vec_model = word2vec.Word2Vec(sentences)
    # word2vec_model.save(path + word2vec_model_path)

    # tokenizer = Tokenizer(num_words=max_features, lower=True)
    # tokenizer.fit_on_texts(datapk)
    # word_index = tokenizer.word_index

    # val_sequences = tokenizer.texts_to_sequences(val_data_pk)
    # val_data_pk = pad_sequences(val_sequences, maxlen=max_length)

    # sequences = tokenizer.texts_to_sequences(datapk)
    # datapk = pad_sequences(sequences, maxlen=max_length)

    '''label prepocessing'''
    # labelpk = to_categorical(labelpk)
    # val_label = to_categorical(val_label_pk)

    '''Embedding'''
    # model = gensim.models.Word2Vec.load(path + word2vec_model_path)
    # word2idx = {"_PAD": 0}
    # vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    #
    # embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    # for i in range(len(vocab_list)):
    #     word = vocab_list[i][0]
    #     word2idx[word] = i + 1
    #     embedding_matrix[i + 1] = vocab_list[i][1]
    # print(embedding_matrix)
    # print(embedding_matrix.shape)

    tokenizer = Tokenizer(num_words=max_features, lower=True)
    tokenizer.fit_on_texts(datapk)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(sequences, maxlen=max_length)
    sequences = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(sequences, maxlen=max_length)
    sequences = tokenizer.texts_to_sequences(val_data_list)
    val_data = pad_sequences(sequences, maxlen=max_length)
    conclude = []
    conclude.append(X_train)
    conclude.append(X_test)
    conclude.append(y_train)
    conclude.append(y_test)
    conclude.append(val_data)
    conclude.append(val_label_pk)
    # conclude.append(embedding_weights)
    write_conclude = open(path + 'cnews.conclude', 'wb')
    pk.dump(conclude, write_conclude)

    '''存储data[0]、label[1]、val_data[2]、val_label[3]、embedding_martix[4]'''
    # store_pk = open(path + pk_path, 'wb')
    # pk_store = []
    # pk_store.append(datapk)
    # pk_store.append(labelpk)
    # pk_store.append(val_data_pk)
    # pk_store.append(val_label_pk)
    # pk_store.append(embedding_matrix)
    # pk.dump(pk_store, store_pk, 0)
    # store_pk.close()
    # print(datapk)
    # print(len(datapk))