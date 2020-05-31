##IMPORTING THE LIBRARY
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from  sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

##READING THE DATA
data=pd.read_csv('mnist.csv')

##VISUALIZING THE HEAD OF THE DATA
data.head()

##SELECTING THE ROW FROM THE DATASET
a=data.iloc[2,1:].values

##DISPLAYING THE SELECTED ELEMENTS
a=a.reshape(28,28)
dtype=np.uint8
plt.imshow(a)

##SEPERATING THE FEATURES AND LABEL
df_x=data.iloc[:,1:] #FEATURES
df_y=data.iloc[:,0]  #LABEL

##PREPARING TEST AND TRAIN SIZE
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)

y_train.head()
##RF CLASSIFIER
rf=RandomForestClassifier(n_estimators=100)

##FIT THE MODEL
rf.fit(x_train,y_train)

##PREDICTION ON TEST DATA
pred=rf.predict(x_test)
pred

##CHECKING THE PREDICTED ACCURACY
s=y_test.values
count=0
for i in range(len(pred)):
  if pred[i]==s[i]:
    count=count+1
    count
##LENGTH OF PREDICTION
len(pred)

##ACCURACY FOR PREDICTION
count/len(pred)
