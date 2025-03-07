import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
from sklearn.linear_model import Lasso
path = os.path.dirname(os.path.abspath(__file__))
path_data = path + '/Input/dataset_after_PCA.csv'
df = pd.read_csv(path_data)
x = df.iloc[0:352, 0:40] 
y = df.iloc[0:352, -1] 
import multiprocessing
n_processes = multiprocessing.cpu_count()

best_mae_lst = []
for seed in range(500):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True) 
        lasso = Lasso(alpha=alpha)
        lasso.fit(xtrain, ytrain)
        y_pred = lasso.predict(x)
        mae = mean_absolute_error(y,y_pred)
        return mae
    

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    with multiprocessing.Pool(processes=n_processes) as pool:
        study.optimize(objective, n_jobs=n_processes, n_trials=100)

    optuna.logging.disable_default_handler()

    best_params = study.best_params
    best_mae = study.best_value
    a = (seed , best_mae , best_params)
    best_mae_lst.append(a)
    print(str(seed+1)+r'/500')

average = sum(item[1] for item in best_mae_lst ) / len(best_mae_lst)
min_tuple = min(best_mae_lst, key=lambda x: x[1])
print("具有最小第二个值的元组:", min_tuple)
print('平均mae:',average)
average = sum(item[1] for item in best_mae_lst) / len(best_mae_lst)
second_values = [item[1] for item in best_mae_lst]
std_deviation = np.std(second_values)
print("标准差:", std_deviation)

lst = []
num = 1
for i in best_mae_lst:
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=i[0])

    lasso = Lasso(alpha=i[2]['alpha'])
    lasso.fit(xtrain, ytrain)
    y_pre_train = lasso.predict(xtrain)
    y_pre_test=  lasso.predict(xtest)
    y_pre = lasso.predict(x)
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
df0 = pd.DataFrame(best_mae_lst,columns=['Seed','MAE','Param'])
df0.to_csv(path+'/Output/res_lambda_parameter.csv',index=False)

df1 = pd.DataFrame(lst,columns=["训练集mae","测试集mae","整体mae","训练集mse","测试集mse","整体mse","训练集r2","测试集r2","整体r2"])
df1.to_csv(path+'/Output/res_lambda_result.csv',index=False,encoding='utf_8_sig')