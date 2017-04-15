import multiprocessing
import numpy as np
from random import shuffle
import tensorflow as tf
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
import csv
import math
def multipl(a, b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab
def corrcoef(x, y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x, y)
    #求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=math.sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den


def rnn(hnum1, hnum2, hnum3):
    file = open(r"wind.csv")
    file.readline()  # 读掉第一行,下次再引用file的时候,将file的文件指针指向第二行开始的文件.
    reader = csv.reader(file)

    raw_data = []
    for date, hors, u, v, ws, wd in reader:
        if ws != 'NA':
            raw_data.append(float(ws))
    # difference = max(raw_data)-min(raw_data)
    # raw_data = [i/difference for i in raw_data]
    # raw_data=[10*math.sin(0.1*i) for i in range(20000)]


    sequence_length = 100  # 代表以往数据的移动窗口宽度 注意取值范围 (可调参数)
    predict_length = 16  # 代表预测数据的移动窗口宽度 即一次预测出的结果数 注意取值范围 (可调参数)
    train_input_all = []
    for i in range(0, len(raw_data[0:-sequence_length - predict_length + 1])):
        temp_list = []
        for j in range(sequence_length):
            temp_list.append([raw_data[i + j]])
        train_input_all.append(temp_list)

    train_label_all = []
    train_label_all1 = []
    for i in range(sequence_length, len(raw_data) - predict_length + 1):
        temp_list = []
        for j in range(predict_length):
            temp_list.append(raw_data[i + j])
        train_label_all.append(temp_list)
        train_label_all1.append(raw_data[i + j])

    seperate_point = 5000  # 测试集与训练集分割点 （可调数）
    test_point = 90000  # 使用的数据量大小（可调数 且必须大于seperate_point）
    test_point_start = 80000
    train_input = train_input_all[0: seperate_point]
    test_input = train_input_all[test_point_start + 1: test_point]
    train_output = train_label_all[0: seperate_point]  # 训练数据标签格式1
    train_output1 = train_label_all1[0: seperate_point]  # 训练数据标签格式2
    test_output = train_label_all[test_point_start + 1: test_point]  # 测试数据标签格式1
    test_output1 = train_label_all1[test_point_start + 1: test_point]  # 测试数据标签格式2
    # 打乱训练集
    index = [i for i in range(len(train_input))]
    shuffle(index)
    train_input = [train_input[index[i]] for i in range(len(index))]
    train_output = [train_output[index[i]] for i in range(len(index))]

    data = tf.placeholder(tf.float32, [None, sequence_length, 1])  # batch_size maxtime deepth
    target = tf.placeholder(tf.float32, [None, predict_length], name='target')
    num_hidden = [hnum1, hnum2, hnum3]  # 隐含层数量(可调参数)
    # cell = rnn_cell.BasicRNNCell(num_hidden)
    # cells = rnn_cell.LSTMCell(num_hidden[0], state_is_tuple=True)
    cell_layer1 = rnn_cell.LSTMCell(num_hidden[0], state_is_tuple=True)
    # cell_layer1 = rnn_cell.DropoutWrapper(cell_layer1, input_keep_prob=0.5, output_keep_prob=0.5)
    cell_layer2 = rnn_cell.LSTMCell(num_hidden[1], state_is_tuple=True)
    # cell_layer2 = rnn_cell.DropoutWrapper(cell_layer2, input_keep_prob=0.5, output_keep_prob=0.5)
    cell_layer3 = rnn_cell.LSTMCell(num_hidden[2], state_is_tuple=True)
    # cell_layer4 = rnn_cell.LSTMCell(num_hidden[3], state_is_tuple=True)
    # cell_layer5 = rnn_cell.LSTMCell(num_hidden[4], state_is_tuple=True)
    cells = rnn_cell.MultiRNNCell([cell_layer1, cell_layer2, cell_layer3])  # 建立多层rnn

    val, state = tf.nn.dynamic_rnn(cells, data, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])

    val_shape = val.get_shape()

    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    last_shape = last.get_shape()
    weight = tf.Variable(tf.truncated_normal([num_hidden[-1], int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.matmul(last, weight) + bias
    prediction_shape = prediction.get_shape()
 

    loss = tf.reduce_mean(tf.square(prediction - target))
    loss_shape = loss.get_shape()
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(loss)

    # mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.square(prediction - target))
    error_sep = tf.square(prediction - target)  # 计算每一个预测分量的误差
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(init_op)  # 在这里，可以执行这个语句，也可以不执行，即使执行了，初始化的值也会被restore的值给override
    # saver.restore(sess, r"parameter_5.ckpt")

    batch_size = 10  # （可调参数）
    no_of_batches = int(len(train_input) / batch_size)
    epoch = 25  # （可调参数）

    total_error1 = 0
    predict_result1 = []
    total_error = 0
    predict_error = []
    predict_result = []
    temp = 0  # 测试变量

    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
            ptr += batch_size
            sess.run(minimize, {data: inp, target: out})
        print("Epoch - ", str(i))

    # sess.run(error, {data: train_input, target: train_output})
    # 观察测试样本的估计情况
    total_error = 0
    predict_error = []
    predict_result = []
    temp = 0
    temp1 = 0
    temp_sep = []
    total_error_sep = []
    for i in range(len(test_input)):
        inp, out = test_input[i:i + 1], test_output[i:i + 1]
        temp1 = sess.run(prediction, {data: inp, target: out})
        temp = sess.run(error, {data: inp, target: out})
        temp_sep = sess.run(error_sep, {data: inp, target: out})
        # print(temp1)
        total_error += temp
        # predict_error.append(temp)
        predict_result.append(temp1[0])
        total_error_sep.append(temp_sep)
    total_error /= len(test_input)
    total_error = math.sqrt(total_error)
    total_error_sep = (np.array(total_error_sep)).mean(axis=0)
    # 观察训练样本的训练情况

    total_error1 = 0
    predict_result1 = []
    temp2 = 0
    temp3 = 0
    temp_sep1 = []
    total_error_sep1 = []
    for i in range(len(train_input)):
        inp, out = train_input[i:i + 1], train_output[i:i + 1]
        temp2 = sess.run(error, {data: inp, target: out})
        temp3 = sess.run(prediction, {data: inp, target: out})
        temp_sep1 = sess.run(error_sep, {data: inp, target: out})
        total_error1 += temp2
        predict_result1.append((temp3[0]))
        total_error_sep1.append(temp_sep1)
    total_error1 /= len(train_input)
    total_error1 = math.sqrt(total_error1)
    total_error_sep1 = (np.array(total_error_sep1)).mean(axis=0)

    # incorrect = sess.run(error, {data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.5f}'.format(i + 1, total_error))
    # print('predict_error')
    # print(predict_error)
    # print('predict_result')
    # print(predict_result)
    saver.save(sess, r"parameter_5.ckpt")
    sess.close()

    # saver = tf.train.Saver()
    # pylab.plot(predict_result)# predict_result1是测试样本的检擦结果predict_result
    # pylab.plot(test_output)
    # pylab.plot(predict_result1)# predict_result1是训练样本的检擦结果predict_result
    # pylab.plot(train_output)#train_output1是训练样本的检查结果test_output1
    corrcoef_result_test = []
    corrcoef_result_train = []
    for cursor in range(16):
        test_output_single = [test_output[i][int(cursor)] for i in range(len(test_output))]
        predict_result_single = [predict_result[i][int(cursor)] for i in range(len(predict_result))]
        corrcoef_result_test.append(corrcoef(test_output_single, predict_result_single))
        #print(corrcoef(test_output_single, predict_result_single))

    for cursor in range(16):
        train_output_single = [train_output[i][int(cursor)] for i in range(len(train_output))]
        predict_result1_single = [predict_result1[i][int(cursor)] for i in range(len(predict_result1))]
        corrcoef_result_train.append(corrcoef(train_output_single, predict_result1_single))
        #print(corrcoef(train_output_single, predict_result1_single))
    '''
    需要记录的数据 1输入训练样本编号 2测试样本编号 3使用的模型类型 4使用的模型参数（隐含层数量 隐含层每层的单元数量）5训练batch大小
    6训练的周期数 7预测结果 8输出误差大小 9输出的相关系数 10训练时间
    '''

    csvfile = open(r'short_result5.csv', 'a')
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'seperate_point', 'test_point_start', 'test_point', 'num_hidden', 'sequence_length',
                     'predict_length', 'batch_size'])
    data = [(epoch, seperate_point, test_point_start, test_point, num_hidden, sequence_length, predict_length,
             batch_size)]
    writer.writerows(data)
    writer.writerow(['corrcoef_result_test'])
    writer.writerow(corrcoef_result_test)
    writer.writerow(['corrcoef_result_train'])
    writer.writerow(corrcoef_result_train)
    writer.writerow(['prediction_result'])
    writer.writerow(['total_error_sep'])
    for i in range(len(total_error_sep)):
        writer.writerow(total_error_sep[i])
    writer.writerow(['total_error'])
    writer.writerow([total_error])
    csvfile.close()

if __name__ == '__main__':
    for i in range(6):
        for j in range(6):
            for k in [4,5]:
                m = multiprocessing.Process(target=rnn, args=((i + 1) * 20, (j + 1) * 20, (k + 1) * 20))
                m.start()
                m.join()
                print(i, j, k)