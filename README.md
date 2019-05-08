这是为keras的使用而编写的自我练习程序
====================================
# 环境依赖：
1. python3.5
2. keras2.2.2
3. numpy1.16.2
4. scikit-learn 0.20.3
5. tensorflow-gpu1.9.0

# 训练集：
1. cnews

# 预处理：
preprocessing_cnews.py
这个文件会读取cnews.train.txt，将每段前的标签取出并排序编号，随后用该编号对应将所有文本按序匹配标签，一并送入sklearn的train_test_split中，得到X_train，X_test,Y_train,Y_test.再用keras的to_categorical。（将其切分为测试集和训练集）  
之后用keras的Tokenizer将文本转换为矩阵与val的验证集汇总，以X_train，X_test,Y_train,Y_test,val_data,val_label的顺序用pickle存储于cnews.conclude文件中



# 实验结果：

方法|f1|recall|precision
----|--------|--------|---------
LSTM（with itself and a flatten)|0.899090938618863|0.9016|0.906488425734453
SimpleRNN(with only itself ）|0.456839776711619|0.4582|0.486761156984287
SimpleRNN(with a Dense）|0.607950788357987|0.6158|0.627004915062812
CNN（with only Conv1D）|0.893030792871051|0.8946|0.897924332106227
CNNadd_convolution1DandMaxPolling1Ddoubled|0.908714193119703|0.9102|0.912187193540087
LSTMaddConvolution1DandMaxPooling1D|0.890570255487257|0.8934|0.902463295185187
