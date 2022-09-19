# Ex.No.1---Data-Preprocessing
## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

## Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

## Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

## Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
```
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train
```
## PROGRAM:
```
import pandas as pd
import numpy as np
df = pd.read_csv("/content/Churn_Modelling.csv")
df.info()
df.isnull().sum()
df.duplicated()
df.describe()
df['Exited'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape
```

## OUTPUT:
### Dataset
![190887932-1be6adad-ee26-4401-b08a-4acb35cac74a](https://user-images.githubusercontent.com/94166007/191054251-ec2ea0fb-3b84-4ba0-aec2-c29f24461acd.png)
### Checking for Null Values
![190887937-e41c748c-2fbf-40cd-bab3-b7cd2ac30e86](https://user-images.githubusercontent.com/94166007/191054300-687229b3-c713-4e8d-99f3-f5cfa209f9aa.png)
### Checking for duplicate values
![190887943-b28d27c3-8840-461f-838b-ff9d21145a30](https://user-images.githubusercontent.com/94166007/191054387-65483cd7-c73c-49e3-9ff4-e8f13486b6f8.png)
### Describing Data
![190887950-7d8ff256-56b5-4dc4-9712-5c512e75d84d](https://user-images.githubusercontent.com/94166007/191054463-f6a609ca-9fd5-496e-90ea-c15d7ada4e6f.png)
![190887955-5df2cb7f-5db3-4aef-a693-8a20d170b900](https://user-images.githubusercontent.com/94166007/191054540-83ca5f6c-a5a8-4514-b5f3-39a82cb67c75.png)

### X - Values
![190887962-7a1747fa-b93c-42be-9aec-8ac5a3f37bda](https://user-images.githubusercontent.com/94166007/191054586-2ae9bde9-14b7-4183-bff0-de28504b1484.png)
### Y - Value
![190887966-c905dfa9-3d4a-48bc-a74b-13cd80c1ba63](https://user-images.githubusercontent.com/94166007/191054645-c06dced8-fb82-4264-ac8e-6dbcb96ba0e9.png)
### X_train values and X_train Size
![190887974-cc92d500-0e84-44c9-b02a-f4a9aae0cfa7](https://user-images.githubusercontent.com/94166007/191054702-85465b76-0283-45ab-a63d-ae14d0027d43.png)
### X_test values and X_test Size
![190887980-468d4264-972c-451e-81bc-710e94b1764f](https://user-images.githubusercontent.com/94166007/191054776-97868fb7-f57d-4570-81a7-b20604f23cc3.png)
### X_train shape
![190887987-14633afc-6cc5-4144-bd37-b7160d9120c2](https://user-images.githubusercontent.com/94166007/191054828-39b297ef-1188-4b4f-8d19-9ec0ce23089f.png)

## RESULT
Data preprocessing is performed in a data set downloaded from Kaggle
