
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#dataproccessing
heartdata = pd.read_csv('"C:\Users\wargo\OneDrive\Desktop\project\heart_disease_data.csv"')

#print(heartdata)
#print(heartdata.info())
#print(heartdata.describe())import numpy as np
#print(heartdata['target'].value_counts())

#1-->Defect
#0-->healthy

x = heartdata.drop(columns='target',axis=1)
y = heartdata['target']

print(y)

#spliting data into training and test data
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
#training
model.fit(x_train,y_train)

#accuraracy on training data
X_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_train)
print('accuracy on training data :',training_data_accuracy)

#accuraracy on test data
X_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(X_test_prediction,y_test)
print('accuracy on training data :',testing_data_accuracy)

#building predictive system
input_data = (64,1,2,125,309,0,1,131,1,1.8,1,0,3)
#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are pridiction for only one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):print('Person does not have heart disease')
else:print('Person has Heart disease')




