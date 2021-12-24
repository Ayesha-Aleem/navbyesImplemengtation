from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
result=[]


learning_rate = 0.01
training_epochs = 1000
display_step = 50

df=pd.read_csv(r'C:\Users\HP\Downloads\drug200.csv')
print(df)
print(type(df.to_numpy()))

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

X = df.drop('Drug', axis=1)
y = df['Drug']




gnb = GaussianNB()
# for training and and testing of a function
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print(X.shape,len(y))
print(X_train.shape,y_train.shape,X_test.shape)

class1 = (y_test != y_pred).sum()
class1 = ((len(y_test) - class1)/len(y_test)) * 100
result.append(class1)

plot_confusion_matrix(gnb, X_test, y_test)  
plt.show()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

print(X.shape,len(y))
class1 = (y_test != y_pred).sum()
class1 = ((len(y_test) - class1)/len(y_test)) * 100
result.append(class1)
print(X_train.shape,y_train.shape,X_test.shape)

clf = svm.SVC()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

print(X.shape,len(y))
class1 = (y_test != y_pred).sum()
class1 = ((len(y_test) - class1)/len(y_test)) * 100
result.append(class1)
print(X_train.shape,y_train.shape,X_test.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

print(X.shape,len(y))
class1 = (y_test != y_pred).sum()
class1 = ((len(y_test) - class1)/len(y_test)) * 100
result.append(class1)
print(X_train.shape,y_train.shape,X_test.shape)

x = ["NaiveBayes", "DecisionTrees", "SupportVectorMachines", "Neural"]
h = result
c = ["red", "green", "orange", "black", "yellow"]
plt.bar(x,h,width=0.3, color=c)
plt.xlabel("Classifications")
plt.ylabel("Percentage")
plt.show()