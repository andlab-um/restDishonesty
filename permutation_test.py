import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn_rvm import EMRVR
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error 
data=pd.read_csv("/Users/huidili/Wu_Lab/honesty_20210203/csv_LOOCV/wholeSFC.csv")
y=data['NaN'].values
print(y.shape)
X=data.drop(['NaN'],axis=1).values

sum=0
for i in range(1000):
    y_shuffle=np.random.permutation(y)
    print(y_shuffle)
    # cross validation 
    loo=LeaveOneOut()
    loo.get_n_splits(X)
    y_actual=[]
    y_predict=[]
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_shuffle[train_index], y_shuffle[test_index]
        X_train_select=np.array([])
        X_test_select=np.array([])
        i=0
        for num, column in enumerate(X_train.T):
            coef,p=pearsonr(column,y_train)
            if p<.01:
                if i==0:
                    X_train_select=np.append(X_train_select,np.reshape(column,(-1,1)))
                    X_test_select=np.append(X_test_select,np.reshape(X_test[:,num],(-1,1)))
                    i+=1
                    np.reshape(X_train_select,(1,-1))
                else:
                    X_train_select=np.c_[X_train_select,np.reshape(column,(-1,1))]
                    X_test_select=np.column_stack([X_test_select,np.reshape(X_test[:,num],(-1,1))])

        X_test_select=np.reshape(X_test_select,(1,-1))
        rvr=EMRVR(kernel="linear")
        rvr.fit(X_train_select, y_train)
        y_p=rvr.predict(X_test_select)
        y_predict.append(y_p[0])
        y_actual.append(y_test[0])

    coef, p=pearsonr(y_actual,y_predict)
    if coef>0.39805532316419573:
        sum+=1
    print(coef)

p_value=sum/1000
print(p_value)
        
    


        
    
