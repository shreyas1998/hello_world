"""	logistic regresssion applied on the iris dataset.
	Basic implementantion done using scikit learn .
	Both one vs one and one vs all methods were used for this multiclass(3 classes) problem using numpy an user defined functions"""

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import collections as coo

# import some data to play with
iris = datasets.load_iris()
X = iris.data # we only take the first two features.
Y = iris.target

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)
							
h = .02  

logreg = linear_model.LogisticRegression(C=1e5)


p0=q0=r0=0

for lab in y_train:					# to store the most occuring class
	if(lab==0):						
		p0=p0+1
	if(lab==1):
		q0=q0+1
	if(lab==2):
		r0=r0+1		
		
res=[]								#One vs One method
for x,y in [(0,1),(1,2),(0,2)]:
	x1_train=[]
	y1_train=[]	
	for a,b in [(x,y)]:
	
		for i in range(len(y_train)):
		
			if(y_train[i]==a or y_train[i]==b):
				x1_train.append(x_train[i])
				y1_train.append(y_train[i])
				
			
	x2_train,x2_test,y2_train,y2_test=train_test_split(x1_train,y1_train,test_size=0.3,random_state=100)	
	logreg.fit(x2_train,y2_train)
	
	
	y_pred=(logreg.predict(x_test))
	res.append(y_pred)
	
b={}	
keys = []
for m1 in zip(res[0],res[1],res[2]):
	
	b=dict(coo.Counter(m1))
	
	if(len(set(b.values()))==1):
			print(q0)									#since q0 is frequency of max frequency class
	key,value=max(b.items(),key=lambda p: p[1])
	keys.append(key)
	
keys = np.array(keys)
for i in range(len(y_test)):

	print("predicted: "+repr(keys[i])+" "+ "actual: "+repr(y_test[i]) )
	
print(100*accuracy_score(keys,y_test))




	






























