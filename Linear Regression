#importing neccesary packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

boston=load_boston()
print(boston.DESCR)

data=boston.data
for name, index in enumerate(boston.feature_names):
  print(index,name)

#assining feature
dataset=data[:,12].reshape(-1,1)

np.shape(data)
np.shape(dataset)

target=boston.target.reshape(-1,1)
np.shape(target)

%matplotlib inline
plt.scatter(dataset, target,color='Blue')
plt.xlabel('population')
plt.ylabel('cost')
plt.show()

#LinearRegression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(dataset,target)
pred=reg.predict(dataset)
#plotting
plt.scatter(dataset, target,color='Blue')
plt.plot(dataset, pred,color='red')
plt.xlabel('population')
plt.ylabel('cost')
plt.show()

#lasso regressor
from sklearn.linear_model import Lasso
re=Lasso()
re.fit(dataset,target)
predi=re.predict(dataset)
#plotting
plt.scatter(dataset, target,color='Blue')
plt.plot(dataset, predi,color='red')
plt.xlabel('population')
plt.ylabel('cost')
plt.show()

#Ridge
from sklearn.linear_model import Ridge
re=Ridge()
re.fit(dataset,target)
predic=re.predict(dataset)
plt.scatter(dataset, target,color='Blue')
plt.plot(dataset, predic,color='red')
plt.xlabel('population')
plt.ylabel('cost')
plt.show()
#polynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
#merging models
from sklearn.pipeline import make_pipeline
model=make_pipeline(PolynomialFeatures(3),reg)
model.fit(dataset, target)
pred=model.predict(dataset)
#plot
plt.scatter(dataset, target,color='Blue')
plt.plot(dataset, pred,color='red')
plt.xlabel('population')
plt.ylabel('cost')
plt.show()
#loss
from sklearn.metrics import r2_score
r2_score(pred,target)
