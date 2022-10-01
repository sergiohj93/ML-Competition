# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
from sklearn import model_selection,base,svm,metrics


#Getting the doughnut feature and the target from the csv and imputing nan 
#values of the feature with the mean.
df = pd.read_csv("Data/train.csv")
y = df["fraudulent"].values
df["required_doughnuts_comsumption"] = df["required_doughnuts_comsumption"].fillna(df["required_doughnuts_comsumption"].mean())
X = df["required_doughnuts_comsumption"].values
#Deleting rows where the target is nan.
idx_nan = np.where(np.isnan(y))
y = np.delete(y,idx_nan)
X = np.delete(X,idx_nan)
X = X.reshape(-1,1)

#Training a svm with a simple train test split...
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=42)
svm_doug = svm.SVC(random_state=42)
svm_doug.fit(X_train,y_train)
yhat = svm_doug.predict(X_test)
#...and obtaining the F1 Score.
f1_svm = metrics.f1_score(y_test,yhat)
print('F1 train 70%: ',f1_svm)
best_f1 = f1_svm
best_svm = copy.deepcopy(svm_doug)

#K-Fold Cross Validation. We choose our best model between the train test split
# one and those of K_Fold.
splits = 3
cv = model_selection.KFold(n_splits = splits,shuffle=True ,random_state=42)
cv.get_n_splits(X)
yhat = np.zeros((X.shape[0],1))
f1_relat = np.zeros(splits)
i = 0
for train_idx, test_idx in cv.split(X):
    X_train,y_train = X[train_idx],y[train_idx]
    X_test,y_test = X[test_idx],y[test_idx]

    svm_doug = svm.SVC(random_state=42)
    svm_doug.fit(X_train,y_train)
    yhat[test_idx] = svm_doug.predict(X_test).reshape(-1,1) 
    f1_relat[i] = metrics.f1_score(y_test, yhat[test_idx])
    if f1_relat[i] > best_f1:
        best_svm = copy.deepcopy(svm_doug)
        best_f1 = f1_relat[i]
    i+=1
f1 = metrics.f1_score(y,yhat)
print('F1 K-folds: ',f1)
print('F1 of the chosen model: ', best_f1)

#Reading the test and predicting it with the best model.
df = pd.read_csv("Data/test.csv")
df["doughnuts_comsumption"] = df["doughnuts_comsumption"].fillna(df["doughnuts_comsumption"].mean())
X = df["doughnuts_comsumption"].values
X = X.reshape(-1,1)
yhat = best_svm.predict(X)
d = {'Id': range(len(yhat)), 'Category': yhat.astype(int)}
df = pd.DataFrame(data=d)
df.to_csv('Data/submission.csv',index=False)
