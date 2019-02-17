import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta  
from datetime import time
import seaborn as sns
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from decimal import *


dataset=pd.read_csv('Hospital1.csv')
dataset=dataset.drop(columns=['Race','CCS Procedure Code','Patient Disposition','Health Service Area','Facility Id','Operating Certificate Number','Facility Name','Discharge Year','Zip Code - 3 digits','Ethnicity','CCS Diagnosis Description','CCS Procedure Description','APR DRG Description','APR MDC Description','APR Severity of Illness Description','APR Severity of Illness Description','Payment Typology 1','Payment Typology 2','Payment Typology 3','Attending Provider License Number','Operating Provider License Number','Other Provider License Number'])
#dataset_obj=dataset["Hospital County","Age Group"].astype('category')
a=dataset.dtypes
dataset1=pd.get_dummies(data=dataset,columns=['Hospital County','CCS Diagnosis Code','APR DRG Code','APR MDC Code','APR Medical Surgical Description'])


dataset1['Age Group'] = LabelEncoder().fit_transform(dataset1['Age Group'])
dataset1['Gender'] = LabelEncoder().fit_transform(dataset1['Gender'])
dataset1['Emergency Department Indicator']=np.where(dataset1['Emergency Department Indicator']=='N',0,1)
dataset1['Abortion Edit Indicator']=np.where(dataset1['Abortion Edit Indicator']=='N',0,1)

dataset1['Type of Admission'].unique()

dataset.dtypes
dataset1['Total Charges']= dataset1['Total Charges'].str.replace('$','')
dataset1['Total Charges']= dataset1['Total Charges'].str.replace(',','')
dataset1['Total Charges']= dataset1['Total Charges'].astype(float)*1.0

#import seaborn as sns
#sns.set(style="whitegrid")
#ax = sns.barplot(x="Type of Admission", y="Total Charges", data=dataset1)

dataset1['Type of Admission'] = dataset1['Type of Admission'].map({'Newborn':0,'Emergency':1,'Not Available':2,'Elective':3,'Urgent':4,'Trauma':5})
dataset1['Type of Admission'].head(6)
    




dataset1['APR Severity of Illness Code'] = LabelEncoder().fit_transform(dataset1['APR Severity of Illness Code'])

dataset1['APR Risk of Mortality'].unique()

ax = sns.barplot(x="APR Risk of Mortality", y="Total Charges", data=dataset1)


dataset1['APR Risk of Mortality'] = dataset1['APR Risk of Mortality'].map({'Minor':0,'Moderate':1,'Major':2,'Extreme':3})
q=dataset1.dtypes

print("Are there missing values? {}".format(dataset1.isnull().any().any()))
qw=dataset1.isnull().sum()

dataset1=dataset1[pd.notnull(dataset1['APR Risk of Mortality'])]
print("Are there missing values? {}".format(dataset1.isnull().any().any()))

dataset1['Length of Stay'].unique()
dataset1['Length of Stay']=dataset1['Length of Stay'].replace(['120 +'],'125')
dataset1['Length of Stay'].dtypes

dataset1['Length of Stay']=dataset1['Length of Stay'].astype(int)
q=dataset1.dtypes




temp=dataset1.groupby([dataset1['Birth Weight']]).agg({'Total Charges':'max'})
temp['Birth Weight'] =temp.index

sns.distplot(temp['Birth Weight'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

dataset1['Birth Weight']=np.where((dataset1['Birth Weight']>=3302) & (dataset1['Birth Weight']<=7427),1,0)

from sklearn.utils import shuffle
dataset1 = shuffle(dataset1)



Y = dataset1.iloc[:500,[9]].values
X = dataset1.iloc[:500,dataset1.columns !='Total Charges'].values


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=23)

print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)


'''
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,random_state=0,criterion='entropy')
classifier.fit(X_train,Y_train)



#predicting the test resutl
 
Y_pred=classifier.predict(X_test)

#make confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import r2_score
r2_score(Y_test, y_pred) 

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, y_pred)








