import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
from sklearn.linear_model import Ridge
path = os.path.dirname(os.path.abspath(__file__))
path_data = path + '/Input/dataset_after_PCA.csv'
df = pd.read_csv(path_data)
x = df.iloc[0:352, 0:40] 
y = df.iloc[0:352, -1] 
import multiprocessing
n_processes = multiprocessing.cpu_count()
from sklearn.linear_model import LinearRegression
lst = []
num = 1
for seed in range(500):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
    lr = LinearRegression(positive=True)
    lr = lr.fit(xtrain,ytrain)
    y_pre_train = lr.predict(xtrain)
    y_pre_test=  lr.predict(xtest)
    y_pre = lr.predict(x)
    #训练集mae
    mae0 = mean_absolute_error(ytrain, y_pre_train)
    #测试集mae
    mae1 = mean_absolute_error(ytest, y_pre_test)
    #整体mae
    mae2 = mean_absolute_error(y, y_pre)
    #训练集mse
    mse0 = mean_squared_error(ytrain, y_pre_train)
    #测试集mse
    mse1 = mean_squared_error(ytest, y_pre_test)
    #整体mse
    mse2 =mean_squared_error(y, y_pre)
    #训练集r2
    r20 = r2_score(ytrain, y_pre_train)
    #测试集r2
    r21 = r2_score(ytest, y_pre_test)
    #整体r2
    r22 =r2_score(y, y_pre)
    element = (mae0,mae1,mae2,mse0,mse1,mse2,r20,r21,r22)
    lst.append(element)
    print(str(num)+r'/500')
    num = num + 1 

df1 = pd.DataFrame(lst,columns=["训练集mae","测试集mae","整体mae","训练集mse","测试集mse","整体mse","训练集r2","测试集r2","整体r2"])
df1.to_csv(path+'/Output/res_lambda_result.csv',index=False,encoding='utf_8_sig')