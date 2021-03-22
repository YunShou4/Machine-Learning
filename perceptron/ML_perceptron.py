import pandas as pd

w_num = 784
data_num = 42000

#设置迭代次数
gen = 10
path = 'D:\File_Yunshou\digit-recognizer\\train.csv'

#将w初始化为0,ALPHA初始化为0
Ws = [0.0 for col in range(w_num)]
ALPHA = [0.0 for col in range(data_num)]
#将b初始化为0，ITA初始化为0.5，
b = 0 
ITA = 0.5
sign = 0

temp = [ ]
DATAs = [ ]
#将数据以二维列表储存
data=pd.read_csv(path)
DATAs = data.values.tolist()



#原始形式
def Training_Original():
    global b, sign, ITA
    for k in range(gen):
        for i in range(data_num):
            #更改训练集中的label列
            if DATAs[i][0] > 0:
                DATAs[i][0] = 1
            else:
                DATAs[i][0] = -1

           # 计算损失函数
            for j in range(w_num):
                sign += DATAs[i][0] * (Ws[j] * DATAs[i][j+1] + b)

            #更新w和b
            if sign <= 0:
                for m in range(w_num):
                    Ws[m] = Ws[m] + ITA * DATAs[i][0] * DATAs[i][m + 1]
                    b = b + ITA * DATAs[i][0]
        print('正在进行第' + str(k+1) + '次迭代')
    print(Ws)
    print(b)



#对偶式
def Training_Dual():
    global b, sign, ITA

    #计算Gram矩阵
    Gram = [ [0.0 for col in range(data_num)] for raw in range(data_num)]
    for i in range(data_num):
         for j in range(data_num):
             for k in range(data_num):
                 Gram[i][j] += DATAs[i][k] * DATAs[j][k]

    for k in range(gen):
        for i in range(data_num):
            #更改训练集中的label列
            if DATAs[i][0] > 0:
                DATAs[i][0] = 1
            else:
                DATAs[i][0] = -1
        
            for j in range(data_num):
                temp[i] += Gram[i][j] * DATAs[j][0]

            # 计算损失函数
            sign += DATAs[i][0] * (ALHPA[i] * temp[i] + b)

            #更新ALHPA和b
            if sign <= 0:
                ALHPA[i] += ITA
                b = b + ITA * DATAs[i][0]
        print('正在进行第' + str(k+1) + '次迭代')
    print(ALPHA)
    print(b)



#模型评价
def TestingPart():
    TP = 0
    FP = 0
    FN = 0
    flag = 0
    for i in range(data_num):
        for j in range(w_num):
            flag += Ws[j] * DATAs[i][j]
        if (flag * DATAs[i][0] > 0) and (DATAs[i][0] > 0):
            TP += 1
        elif (flag * DATAs[i][0] > 0) and (DATAs[i][0] <= 0):
            FP += 1
        elif (flag * DATAs[i][0] <= 0) and (DATAs[i][0] > 0):
            FN += 1
    print('精确率为：' + str(TP / (TP + FP)))
    print('召回率为：' + str(TP / (TP + FN)))



Training_Original()
TestingPart()
#Training_Dual()