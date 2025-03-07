import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import optuna
import matplotlib.pyplot as plt


# path = os.path.dirname(os.path.abspath(__file__))
# path_data = path + '/data.csv'
# df = pd.read_csv(path_data)



# x = df.drop(columns=['material_id','Cohesive_Energy','Index'])
# x = preprocessing.scale(x)
# y = df.iloc[:, -1]

# from sklearn.ensemble import RandomForestRegressor
# lst = []
# for seed in range(2000):
#     xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
#     rf = RandomForestRegressor(n_estimators=10,random_state = seed)
#     rf.fit(xtrain, ytrain)
#     y_pre_train = rf.predict(xtrain)
#     y_pre_test=  rf.predict(xtest)
#     y_pre = rf.predict(x)
#     #训练集mae
#     mae0 = mean_absolute_error(ytrain, y_pre_train)
#     #测试集mae
#     mae1 = mean_absolute_error(ytest, y_pre_test)
#     #整体mae
#     mae2 = mean_absolute_error(y, y_pre)
#     #训练集mse
#     mse0 = mean_squared_error(ytrain, y_pre_train)
#     #测试集mse
#     mse1 = mean_squared_error(ytest, y_pre_test)
#     #整体mse
#     mse2 =mean_squared_error(y, y_pre)
#     #训练集r2
#     r20 = r2_score(ytrain, y_pre_train)
#     #测试集r2
#     r21 = r2_score(ytest, y_pre_test)
#     #整体r2
#     r22 =r2_score(y, y_pre)
#     m = (r20,r21,r22)
#     lst.append(m)

# import csv

# def write_tuples_to_csv(lst, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(lst)


# write_tuples_to_csv(lst, 'D:\BaiduSyncdisk\机器学习\图片最终版\第三张图-机器学习\SHAP Value和gini-importance图\output10.csv')

# from sklearn.ensemble import RandomForestRegressor
# import multiprocessing
# n_processes = multiprocessing.cpu_count()
# lst = []

# best_mae_lst = []
# for seed in range(1095,1096):
#     xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
#     def objective(trial):
#         n_estimators = trial.suggest_int('n_estimators',5,100)
#         max_depth = trial.suggest_int('max_depth', 5,100)
        
#         rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state = seed)
#         rf.fit(xtrain, ytrain)
#         # y_pre_train = rf.predict(xtrain)
#         y_pre_test=  rf.predict(xtest)
#         # y_pre = rf.predict(x)
#         mae = -r2_score(ytest, y_pre_test)
#         # mae =-r2_score(y, y_pre)
#         return mae

#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_jobs=n_processes, n_trials=500)
#     optuna.logging.disable_default_handler()
#     best_params = study.best_params
#     best_mae = study.best_value
#     a = (seed , best_mae , best_params)
#     best_mae_lst.append(a)
#     print(str(seed+1)+r'/1000')
# average = sum(item[1] for item in best_mae_lst ) / len(best_mae_lst)
# min_tuple = min(best_mae_lst, key=lambda x: x[1])
# print("具有最小第二个值的元组:", min_tuple)
# print('平均mae:',average)
# average = sum(item[1] for item in best_mae_lst) / len(best_mae_lst)
# second_values = [item[1] for item in best_mae_lst]
# std_deviation = np.std(second_values)
# print("标准差:", std_deviation)
# lst = []
# num = 1
# for i in best_mae_lst:
#     xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=i[0])

#     rf = RandomForestRegressor(n_estimators=i[2]['n_estimators'], max_depth=i[2]['max_depth'],random_state=i[0])
#     rf.fit(xtrain, ytrain)
#     y_pre_train = rf.predict(xtrain)
#     y_pre_test=  rf.predict(xtest)
#     y_pre = rf.predict(x)
#     #训练集mae
#     mae0 = mean_absolute_error(ytrain, y_pre_train)
#     #测试集mae
#     mae1 = mean_absolute_error(ytest, y_pre_test)
#     #整体mae
#     mae2 = mean_absolute_error(y, y_pre)
#     #训练集mse
#     mse0 = mean_squared_error(ytrain, y_pre_train)
#     #测试集mse
#     mse1 = mean_squared_error(ytest, y_pre_test)
#     #整体mse
#     mse2 =mean_squared_error(y, y_pre)
#     #训练集r2
#     r20 = r2_score(ytrain, y_pre_train)
#     #测试集r2
#     r21 = r2_score(ytest, y_pre_test)
#     #整体r2
#     r22 =r2_score(y, y_pre)

#     element = (mae0,mae1,mae2,mse0,mse1,mse2,r20,r21,r22)
#     lst.append(element)
#     print(str(num)+r'/1000')
#     num = num + 1 

#     print(r20)
#     print(r21)

# df0 = pd.DataFrame(best_mae_lst,columns=['Seed','MAE','Param'])
# df0.to_csv(path+'/Output/res_lambda_parameter.csv',index=False)

# df1 = pd.DataFrame(lst,columns=["训练集mae","测试集mae","整体mae","训练集mse","测试集mse","整体mse","训练集r2","测试集r2","整体r2"])
# df1.to_csv(path+'/Output/res_lambda_result.csv',index=False,encoding='utf_8_sig')


path = os.path.dirname(os.path.abspath(__file__))
path_data = path + '/data.csv'
df = pd.read_csv(path_data)



x = df.drop(columns=['material_id','Cohesive_Energy','Index'])
columns = x.columns
x = preprocessing.scale(x)
x = pd.DataFrame(x, columns= columns)

y = df.iloc[:, -1]

from sklearn.ensemble import RandomForestRegressor
seed = 1095
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
rf = RandomForestRegressor(n_estimators=24, max_depth=12 ,random_state = seed)
rf.fit(xtrain, ytrain)

y_pre_train = rf.predict(xtrain)
y_pre_test=  rf.predict(xtest)
y_pre = rf.predict(x)
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

import shap
# 初始化SHAP解释器
explainer = shap.TreeExplainer(rf)
# 计算SHAP值

shap_values = explainer.shap_values(x)

