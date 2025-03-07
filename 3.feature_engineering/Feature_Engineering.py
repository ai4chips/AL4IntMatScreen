
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns
import matplotlib
dataset = pd.read_csv(".\\Origin_descriptors.csv", encoding='gbk')
dataset_ = dataset.drop(columns=["formula_pretty","material_id"])
column_lable = []
for i in range(np.shape(dataset_)[1]):
    std = np.std(dataset_.iloc[:,i])
    column = dataset_.columns.tolist()
    # print("第",i,"列的std为",std)
    if std == 0 :
        column_lable.append(column[i])

dataset_ = dataset_.drop(columns=column_lable)

# dataset_.to_csv(".\\descriptors_varience_not_zero.csv",index=None)

X = preprocessing.scale(dataset_)

X_df = pd.DataFrame(X,columns=dataset_.columns)

fa = FactorAnalyzer(n_factors=40,rotation="varimax", method="principal", use_smc=True)
fa.fit(X_df)
ev,v = fa.get_eigenvalues()
X_df_f = fa.transform(X_df)
result =pd.DataFrame(X_df_f)
result.to_csv(".\\descriptors_after_PCA.csv", encoding='utf_8_sig', na_rep='None',index = None)

