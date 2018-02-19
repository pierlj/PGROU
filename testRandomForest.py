from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy
import matplotlib.pyplot as plt

def importData(chemin):
    file=open(chemin,'r')
    data=file.readlines()
    names=[]
    X=[]
    y=[]
    for i in data:
        tab=i.split(" ")
        names.append(tab[0])
        X.append(tab[1:-1])
        y.append(tab[-1][:-1])
    
    for i in range (len(X)):
        for j in range (len (X[i])):
            X[i][j]=float(X[i][j])
    file.close()
    return(names,X,y)

def validation(clf,X,y):
    #print("Prediction:")
    pred=clf.predict(X)
    confusionMatrix=confusion_matrix(pred,y)
    #print(confusionMatrix)
    precision=sum(numpy.diagonal(confusionMatrix))/sum(sum(confusionMatrix))
    return precision
    


(names1,X1,y1)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\data.csv")
(names2,X2,y2)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\test.csv")



p1=[]
p2=[]
x=numpy.array([i for i in range(1,10)])
clf1 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features='log2')
clf1.fit(X1, y1)
clf2 = RandomForestClassifier(n_estimators=100,criterion='entropy')
clf2.fit(X2, y2)
j=4
for i in range(2,30):
    
    
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i, j), random_state=1)
    clf.fit(X1,y1)
    clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i, j), random_state=1)
    clf1.fit(X1,y1)

    p1.append(validation(clf,X2,y2))
    p2.append(validation(clf1,X1,y1))


plt.plot(p1)
plt.plot(p2)
print(numpy.mean(p1))
print(max(p1))
print(min(p1))
print("\n")
print(numpy.mean(p2))
print(max(p2))
print(min(p2))
    
plt.show()


