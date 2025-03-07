import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os

path = os.path.dirname(os.path.abspath(__file__))
path_data = path + '/Input/dataset_after_PCA.csv'
df = pd.read_csv(path_data)
x = df.iloc[0:352, 0:40] 
y = df.iloc[0:352, -1] 
from sklearn.ensemble import RandomForestRegressor
import multiprocessing
n_processes = multiprocessing.cpu_count()
lst = []

seed = 540

xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.1,random_state=seed)
rf = RandomForestRegressor(n_estimators=36, max_depth=17 ,random_state = seed)
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


print(mae0,mae1,mae2)

print(mse0,mse1,mse2)

print(r20,r21,r22)




# x_left = df.iloc[352:, 0:40] 

# y_left = rf.predict(x_left)

# x_left = df.iloc[352:, 0:42] 

# x_left.to_csv('D:\BaiduSyncdisk\机器学习\图片最终版\第三张图-机器学习\不同轮次对比图\第一轮\\x_left.csv', index=False)

# # 创建一个DataFrame来保存预测结果
# predictions = pd.DataFrame({'Prediction': y_left})

# # 将DataFrame保存为CSV文件
# predictions.to_csv('D:\BaiduSyncdisk\机器学习\图片最终版\第三张图-机器学习\不同轮次对比图\第一轮\predictions.csv', index=False)

