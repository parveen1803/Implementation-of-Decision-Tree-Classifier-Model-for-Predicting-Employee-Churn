# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy score of the model.
6. Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Parveen Fathima M
RegisterNumber: 212219040103  
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project",
"average_montly_hours","time_spend_company","Work_accident",
"promotion_last_5years","salary"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
acc
dt.predict([[.5,.8,9,260,6,0,1,2]])
```

## Output:
## Initial Dataset:
![initial](https://user-images.githubusercontent.com/87666371/174470265-a7b49785-c1ab-47fd-8298-739a423cf4cb.png)

## Dataset Information:
![datainfo](https://user-images.githubusercontent.com/87666371/174470283-8eda76e9-b8e8-4746-8759-bb4805910764.png)

## Null dataset:
![nulldataset](https://user-images.githubusercontent.com/87666371/174470306-a53d54db-bd91-49de-8c67-f722e4bb1c13.png)

## Value counts in left column:
![valcount](https://user-images.githubusercontent.com/87666371/174470329-bae0b69c-9b61-41d9-8570-66eedf316784.png)

## Encoded dataset:
![encodeddata](https://user-images.githubusercontent.com/87666371/174470349-5f453843-a1e4-4ff5-8ed1-04d95172db85.png)

## x set:
![xdataset](https://user-images.githubusercontent.com/87666371/174470361-e0138358-9ab1-48aa-bdbf-d87b8765ecf3.png)

## y values:
![yvalues](https://user-images.githubusercontent.com/87666371/174470368-862a88ed-40d2-40c0-9a08-1538a5892a9e.png)

## Accuracy Score:
![accuracysco](https://user-images.githubusercontent.com/87666371/174470391-66ee857d-c2f5-45a5-b236-c03443a5bdc1.png)

## Dataset Prediction:
![datapred](https://user-images.githubusercontent.com/87666371/174470427-08af7f4c-b606-4197-89ff-86d6b419a80f.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
