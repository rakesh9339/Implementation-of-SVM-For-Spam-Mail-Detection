# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1)Collect a labeled dataset of emails, distinguishing between spam and non-spam.
    
2)Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.
    
3)Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.
  
4)Split the dataset into a training set and a test set.
    
5)Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.
    
6)Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.
    
7)Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.
    
8)Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.
    
9)Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: RAKESH JS
RegisterNumber: 212222230115
```
``` PYTHON
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### 1. Result output
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/c08810ba-4b33-43e0-86b9-88e149733769)

### 2. data.head() 
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/06725080-0126-4ff8-8a80-bb89d4c3addc)

### 3. data.info()
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/0bdf2600-0403-4357-a19b-411728f33895)

### 4. data.isnull().sum()
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/9088861a-060c-4063-9a9b-aac56267da47)

### 5. Y_prediction value
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/f05da199-a5b1-49a4-ab2c-e431e481082d)

### 6. Accuracy value
![image](https://github.com/aldrinlijo04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118544279/ba64d866-04f4-48d6-8d5b-fb253524473a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
