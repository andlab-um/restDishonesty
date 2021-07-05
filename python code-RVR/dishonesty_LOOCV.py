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
X=data.drop(['NaN'],axis=1).values
select_number=[0]*34716

loo=LeaveOneOut()
loo.get_n_splits(X)
y_actual=[]
y_predict=[]
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train_select=np.array([])
    X_test_select=np.array([])
    i=0
    for num, column in enumerate(X_train.T):
        coef,p=pearsonr(column,y_train)
        if p<.01:
            select_number[num]+=1
            if i==0:
                X_train_select=np.append(X_train_select,np.reshape(column,(-1,1)))
                X_test_select=np.append(X_test_select,np.reshape(X_test[:,num],(-1,1)))
                i+=1
                np.reshape(X_train_select,(1,-1))
            else:
                X_train_select=np.c_[X_train_select,np.reshape(column,(-1,1))]
                X_test_select=np.column_stack([X_test_select,np.reshape(X_test[:,num],(-1,1))])

    X_test_select=np.reshape(X_test_select,(1,-1))
    X_train_select=np.reshape(X_train_select,(28,-1))
    print(X_train_select.shape)
    rvr=EMRVR(kernel="linear")
    rvr.fit(X_train_select, y_train)
    y_p=rvr.predict(X_test_select)
    y_predict.append(y_p[0])
    y_actual.append(y_test[0])


coef, p=pearsonr(y_actual,y_predict)
print(coef)
print(p)
if p<.001:
    p='<.001'
else:
    p='='+str(round(p,3))

coef='='+str(round(coef,2))


print(mean_squared_error(y_actual,y_predict))
select_percent=[x/29 for x in select_number]
robust_edge=[]
robust_num=0
for i in range(len(select_percent)):
    if select_percent[i] >0.9:
        robust_edge.append(i+1)
        robust_num+=1
print("robust edges are",robust_edge)
print("robust number is",robust_num)
        
        
    
df=pd.DataFrame({'actual dishonesty rate':y_actual, 'predicted dishonesty rate':y_predict})
sns.set()
sns.set_style("white")
sns.set_style("ticks")
ax=sns.regplot(x="actual dishonesty rate", y="predicted dishonesty rate", data=df,scatter_kws = {'color': '#e5c79b', 'alpha': 0.8},line_kws={'label':"r: {0}\np: {1}".format(coef,p),"color":"#C38427","alpha":1,"lw":2}).legend(loc="best")
# palette=sns.color_palette("Paired")
sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
# sns.palplot(palette)
plt.show()
    












