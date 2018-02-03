from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy

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
    print("Prediction:")
    pred=clf.predict(X)
    confusionMatrix=confusion_matrix(pred,y)
    print(confusionMatrix)
    precision=sum(numpy.diagonal(confusionMatrix))/sum(sum(confusionMatrix))
    print(precision)
    


(names1,X1,y1)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\data.csv")
(names2,X2,y2)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\test.csv")


clf1 = RandomForestClassifier(n_estimators=len(X1))
clf1.fit(X1, y1)

clf2 = RandomForestClassifier(n_estimators=len(X2))
clf2.fit(X2, y2)

validation(clf1,X2,y2)
validation(clf2,X1,y1)



