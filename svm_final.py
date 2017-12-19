"""	support vector machine algorithm on the iris datasets
	The implementation was done using scikit learn.
	Three kernels -linear,polynomial and rbf kernel was used and accuracy was calculated"""

from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np
from sklearn.cross_validation import train_test_split

iris = load_iris()


#load data
x= iris.data
y= iris.target

x,x_test,y,y_test=train_test_split(x,y,test_size=0.2)

kernels=["linear", "poly", "rbf"]					#list containg the aforementioned kernels


for k in range(3):
	count=0
	clf = svm.SVC(kernel=kernels[k])
	clf.fit(x,y)
	print("\nkernel %s \n" %kernels[k])
	
	for i in range(len(y_test)):
		prediction=clf.predict([x_test[i]])
		print("predicted:"+repr(int(clf.predict([x_test[i]])))+" "+"actual:"+ repr(y_test[i]))
		
		if(prediction==y_test[i]):
			count+=1
	accuracy=((count*100)/len(y_test))		
	print("accuracy %f percent"%accuracy)		
