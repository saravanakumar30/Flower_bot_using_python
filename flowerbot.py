import pandas as pd
import numpy as np

from sklearn.datasets import load_iris #sklearn-framework,dataset-package,load_iris-functions
iris_dataset=load_iris()# all the function in load_iris is now stored in iris_dataset

# print("Keys:{}".format(iris_dataset.keys()))

#here iris_dataset is in dictionary datatype so it has key and value
#to see the keys in iris_dataset
#(iris_dataset.keys()) this function used

#in dictionary all the keys has vales.
#now we going to see all the values in  keys one by one

#here we see the Data values
# print("type of Data{}\n".format(type(iris_dataset['data'])))#datatype
# print("shape of Data{}\n".format(iris_dataset['data'].shape))#shape - rows and coulmns
# print("size of Data{}\n".format(iris_dataset['data'].size))# size - num of datas
# print("First five rows Data:\n{}".format(iris_dataset['data'][:]))#print data with slice fun()

# print("Target:{}\n".format(iris_dataset['target']))
#here we see the target values
#0 is setosa
#1 is versicolor
#2 is Virginica

# print("Target_names:{}\n".format(iris_dataset['target_names']))
#here we see the targetnames values

val=iris_dataset['DESCR']
#start_val=val[:] #slicing
# print(val)
#here we see the Descr values
#descr means description
#overall discription of dataset

# print("Feature Names:{}\n".format(iris_dataset['feature_names']))
#here we see the Descr values


# print("File Name{}\n".format(iris_dataset['filename']))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0
)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)


X_new=np.array([[5,2.9,1,0.2]])
#X_new=np.array()
#print("X_new Shape:{}".format(X_new.shape))

prediction=knn.predict(X_new)
#print("Prediction:{}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


y_pred=knn.predict(X_test)
#print("Test Predicted Data: {}".format(y_pred))
#print(len(y_pred))

#print("Test Set Score:{}".format(np.mean(y_pred==y_test)))