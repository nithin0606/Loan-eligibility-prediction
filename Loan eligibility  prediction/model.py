import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("loandata.csv")
df.head()

df.isnull().sum()
df=df.dropna()

df=df.drop(['Loan_ID'],axis=1)

df.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

df['Dependents'].value_counts()

df=df.replace(to_replace='3+', value=4)

sns.countplot(x='Education',hue='Loan_Status',data=df)
sns.countplot(x='Married',hue='Loan_Status',data=df)

df.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
            'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

x=df.drop(columns=['Loan_Status'],axis=1)
y=df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y,random_state=2)
x.shape, X_train.shape, X_test.shape

classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

X_train_pred=classifier.predict(X_train)
acc=accuracy_score(X_train_pred,y_train)

print("Accuracy:", acc)

pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(classifier.predict([[1,1,2,1,1,5417,4196.0,267.0,360.0,1.0,2]]))