# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
### Program to implement the simple linear regression model for predicting the marks scored.
### Developed by:Jwalamukhi S
### RegisterNumber:212223040079 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
1.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/711522f6-2621-4ba3-ab82-b6efbe69f4ab)
2.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/633b54a6-a01b-4a4d-a481-d4fb2d914510)
3.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/7e5414c5-5223-4823-81e2-aef9a42de629)
4.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/6c34451f-47d9-4239-8a40-42548673b76c)
5.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/44f3f01b-0ab2-4f57-ab14-fb128029d5ce)
6.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/9440cc6c-0051-469f-87a5-3bcc8e5b3741)
7.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/1f090553-b84b-4a74-ac3d-21b62ac7c742)
8.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/03e17786-7e2e-4b65-8e25-d6822caef078)
9.  ![image](https://github.com/Jwalamukhi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953628/2d5404d4-9d1a-497f-89d4-4f16eff9010d)



## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
