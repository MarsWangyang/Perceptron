#!/usr/bin/env python
# coding: utf-8

# In[3]:


#code import
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import random
import math
import os

window = tk.Tk()
window.title('Perceptron')
window.geometry('600x700')

def sig(v):
    if v>=0:
        return 1
    else:
        return 0
    
#ListBox
#select file
def file_selection():
    value = lb.get(lb.curselection())                        #滑鼠選到的檔案稱value
    var4.set(value)    
    raw_data(value)
    
#活化函數

#re-decide the output 
def raw_data(value):
    learning_rate_percent = float(var1.get())
    learning_rate = learning_rate_percent / 100
    iteration = int(var2.get())
    convergence_training_percent = float(var3.get())
    convergence_training = convergence_training_percent / 100
    data = []
    with open(value ,'r', encoding = 'utf8' ) as f:
        for line in f:
            data.append([float(ele) for ele in line.split()])
        small = data[0][2]                                       #define the original value
        for i in range(0,len(data),1):                           #if data[i][2] < oringinal value,
            if data[i][2] < small:                               #data[i][2] will become the new 'small'
                small = data[i][2]                               #Then,we can find the smaller one to be zero.
        for i in range(0,len(data),1):     
            if data[i][2] == small:                              #if this data[i][2] ==small,
                data[i][2] = 0                                   #we assign the data to be the 'zero'
            else:                                                #if it not,
                data[i][2] = 1                                   #we assign the data to be the 'one.'
        random.shuffle(data)                                     #先將原始資料打亂隨機排列,再分割成為testing_data,training_data.                                                 
        train(data,convergence_training,learning_rate,iteration)
        return data                                              #把data值return至定義函數中
#training:  
def train(data,convergence_training,learning_rate,iteration):
    training_data = []                               #categorize traing_data & testing_data
    testing_data = []
    for i in range(0,int(len(data)*(2/3))):
        training_data.append(data[i])

    for x in range(int(len(data)*(2/3)),len(data)):
        testing_data.append(data[x])

    a = np.array(training_data,dtype=np.float)     #prepare for splilting array
    b = np.array(testing_data,dtype=np.float)      #turn into the type of array
    c = np.full(len(training_data),-1)
    d = np.full(len(testing_data),-1)

    training_data_x = a[:,:2]
    training_data_y = a[:,2]
    testing_data_x = b[:,:2]
    testing_data_y = b[:,2]

    input_training_data = np.c_[c,training_data_x] #add a column of -1
    input_testing_data = np.c_[d,testing_data_x]    

    w1 = random.uniform(-1,1)
    w2 = random.uniform(-1,1)
    weight = np.c_[-1,w1,w2]

    right = 0
    fail = 0
    counter = 0
    training_accuracy = 0.00
    while (training_accuracy < convergence_training) and (counter < iteration) :
        for i in range(0,len(training_data)):                                        #train my training_data for specific weight
            v = np.dot(weight,input_training_data[i])                                #inner product
            y = sig(v) 
            if(y != training_data_y[i]):
                fail = fail + 1
                if(y == 0):
                    weight = weight + learning_rate * input_training_data[i]
                else:
                    weight = weight - learning_rate * input_training_data[i]
            else:
                right = right + 1
        counter += 1
        training_accuracy = (right/(right+fail))*100
    var5.set(training_accuracy)
    var7.set(weight)
    test(data,weight,testing_data,testing_data_y,input_testing_data,learning_rate,iteration)

def test(data,weight,testing_data,testing_data_y,input_testing_data,learning_rate,iteration):
    counter = 0
    right = 0
    fail = 0
    testing_accuracy = 0.00
    while(counter < iteration):
        for i in range(0,len(testing_data)):
                v = np.dot(weight,input_testing_data[i])
                y = sig(v)
                if(y != testing_data_y[i]):
                    fail = fail + 1
                else:
                    right = right + 1
        counter += 1
        testing_accuracy = (right/(right+fail))*100
    var6.set(testing_accuracy)
    drawing(data,weight)

def drawing(data,weight):

    x = np.linspace(-10,10,100)

    a = weight[0][1]                        #w1                     #w1*x1
    b = weight[0][2]                        #w2                     #w2*x2
    c = weight[0][0]                        #w0                     #w0*(x0=-1)

    y_perception = (c-a*x)/b                                        #draw the line of perception

    plt.plot(x ,y_perception ,c = 'm' ,label = 'perceptron')

    for i in range(0,len(data)):
        if data[i][2] == 1 :
            data_marker = "o"
            data_color = 'b'
        else:
            data_marker = "o"
            data_color = 'r'

        plt.plot(data[i][0],data[i][1],c = data_color,marker = data_marker)

    plt.title("perceptron")
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    plt.legend(loc = 'best')
    plt.show()
    
#GUI
#Label&Entry
var1 = tk.StringVar()
L1 = tk.Label(window,text = '學習率learning rate(%) = ').place(x = 120, y = 350)
entry_learning = tk.Entry(window,textvariable = var1)
entry_learning.place(x = 300, y = 350 )

var2 = tk.StringVar()
L2 = tk.Label(window,text = '疊代次數iteration(次)    = ').place(x = 120, y = 400)
entry_iteration = tk.Entry(window,textvariable = var2)
entry_iteration.place(x = 300 , y = 400)

var3 = tk.StringVar()
L3 = tk.Label(window,text = '訓練成功率accurate rate(%) = ').place(x = 120, y = 450)
entry_accurate_rate = tk.Entry(window,textvariable = var3)
entry_accurate_rate.place(x = 300 , y = 450)

var4 = tk.StringVar()
L4 = tk.Label(window,text = '您所選擇的檔案 ： ').place(x = 330, y = 175 )
entry_choose = tk.Entry(window,textvariable = var4)
entry_choose.place(x = 330 , y = 200)
    
L_input = tk.Label(window, text = '請選擇您要訓練的檔案：')
L_input.place(x = 120 , y = 80)
var_data = tk.StringVar()
var_data.set(('2Ccircle1.txt','2Circle1.txt','2Circle2.txt','2CloseS2.txt','2CloseS3.txt',
            '2cring.txt','2CS.txt','2Hcircle1.txt','2ring.txt','perceptron1.txt','perceptron2.txt'))
lb = tk.Listbox(window, listvariable = var_data, height = 11)
lb.place(x = 120 , y = 100)

var5 = tk.StringVar()
L5 = tk.Label(window,text = ' 訓練辨識率training accuracy(%) = ').place(x = 120 , y = 550)
entry_5 = tk.Entry(window,textvariable = var5)
entry_5.place(x = 320 , y = 550)

var6 = tk.StringVar()
L6 = tk.Label(window,text = ' 測試辨識率testing accuracy(%)  = ').place(x = 120 , y = 600)
entry_6  = tk.Entry(window,textvariable = var6 )
entry_6.place(x = 320 , y = 600)

var7 = tk.StringVar()
L7 = tk.Label(window,text = ' 鍵結值weight                                =').place(x = 120 , y = 650)
entry_7 = tk.Entry(window,textvariable = var7 , width = 30)
entry_7.place(x = 320 ,y = 650)

L_result = tk.Label(window , text = ' 訓練結果 : ')
L_result.place(x = 100 , y = 500)

#button
b_go = tk.Button(window,text = 'Go' ,bg = 'green' , width = 15 ,height = 5 , command= file_selection )
b_go.place(x = 480 , y = 370)

window.mainloop()        
        


