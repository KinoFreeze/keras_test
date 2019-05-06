import pickle as pk


def read_file(path):
    read = open(path, 'r')
    content = read.read()
    read.close()
    return content


def write_file(content, path):
    write = open(path, 'w')
    write.write(''.join(content))
    write.close()


if __name__ == '__main__':
    read_path = '/home/wzc/data/2019-3-19/cnews.train.txt'
    write_label_path = '/home/wzc/data/2019-3-19/cnews_label.txt'
    write_data_path = '/home/wzc/data/2019-3-19/cnews_data.txt'
    write_word_num_path = '/home/wzc/data/2019-3-19/cnews_word_num.txt'
    write_label2id_path = '/home/wzc/data/2019-3-19/cnews_label2id.txt'

    read_val_path = '/home/wzc/data/2019-3-19/cnews.val.txt'
    write_val_path = '/home/wzc/data/2019-3-19/cnews_val.pkl'

    write_label_pk = '/home/wzc/data/2019-3-19/cnews_label.pkl'
    write_data_pk = '/home/wzc/data/2019-3-19/cnews_data.pkl'

    label = []
    data = []

    read = open(read_path, 'r')
    for item in read.readlines():
        label.append(item[:2])
        data.append(item[3:])

    # write_label = open(write_label_path, 'a', encoding="utf-8")
    # write_data = open(write_data_path, 'a', encoding="utf-8")
    # # write_word_num = open(write_word_num_path, 'a', encoding="utf-8")
    # write_label2id = open(write_label2id_path, 'a', encoding="utf-8")
    # read_label2id = open(write_label2id_path, 'r', encoding="utf-8")

    write_val_pk = open(write_val_path, 'wb')
    # write_label_pk = open(write_label_pk, 'wb')
    # write_data_pk = open(write_data_pk, 'wb')

    '''label2id的存储'''
    # label2id = {}
    # count = 0
    # for item in label:
    #     if item not in label2id:
    #         label2id[item] = count
    #         write_label2id.write(''.join(str(count)))
    #         write_label2id.write(''.join('\t'))
    #         write_label2id.write(''.join(item))
    #         write_label2id.write(''.join('\n'))
    #         count += 1
    # print(label2id)
    # write_label2id.close()

    '''label标签的存储'''
    # label2id = {}
    # for item in read_label2id.readlines():
    #     items = item.strip().split('\t')
    #     print(items)
    #     label2id[items[1]] = items[0]
    # label_list = []
    # labels = []
    # for item in label:
    #     labels.append(label2id.get(item))
    # pk.dump(labels, write_label_pk, 0)


    #     write_label.write(label2id.get(item))
    #     write_label.write('\n')
    # write_label.close()

    '''data数据文本的存储'''
    # data_list = []
    # for item in data:
    #     for temp in item:
    #         temps = str(temp) + " "
    #         data_list.append(temps)
    # del data_list[len(data_list) - 1]
    # write_data.writelines(''.join(data_list))
    # write_data.close()
    #
    # datapk = []
    # for item in data:
    #     datapk.append(item)
    # # print(datapk)
    # pk.dump(datapk, write_data_pk)


    '''data中字符编号'''
    # word = []
    # for item in data:
    #     for temp in item:
    #         if temp not in word and temp != '\n' and temp != '\t' and temp != ' ':
    #             word.append(temp)
    #             # write_word_num.write(''.join(temp))
    # for i, item in enumerate(word):
    #     result = str(i + 1) + '\t' + str(item) + '\n'
    #     write_word_num.write(result)
    # write_word_num.close()




    '''val'''
    read = open(read_val_path, 'r')
    for item in read.readlines():
        label.append(item[:2])
        data.append(item[3:])
    valpk = []
    for item in data:
        valpk.append(item)
    pk.dump(valpk, write_val_pk)