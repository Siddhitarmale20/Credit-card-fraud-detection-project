import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('creditcard.csv')
df

df.isnull().sum()

df.info()

non_fraud=len(df[df.Class==0])

fraud=len(df[df.Class==1])

df['Class'].value_counts()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

df['Normalized_amount']=scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Amount'],inplace=True ,axis=1)
df.describe()

x=df.drop(['Class'],axis=1)
y=df.Class

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

len(x_train)
len(y_test)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()

reg.fit(x_train,y_train)

reg.predict(x_test)

reg.score(x_train,y_train)
reg.score(x_test,y_test)

