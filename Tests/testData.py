from sklearn.preprocessing import StandardScaler
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
    return(names,numpy.array(X),numpy.array(y))
    
    
(names1,X1,y1)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\data.csv")
(names2,X2,y2)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\test.csv")


scaler = StandardScaler()
scaler.fit(X1)
mean1=scaler.mean_
var1=scaler.var_
scaler.fit(X2)
mean2=scaler.mean_
var2=scaler.var_

print(mean1-mean2)
print(numpy.linalg.norm(mean1-mean2))
print(numpy.linalg.norm(var1-var2))
