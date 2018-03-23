from sklearn.preprocessing import StandardScaler
import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm

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

print(numpy.mean(mean1))
print(numpy.linalg.norm(mean1-mean2))
print(numpy.linalg.norm(var1-var2))

x_axis = numpy.arange(0.4, 0.7, 0.001)
# Mean = 0, SD = 2.
for i in range(30):
    plt.figure()
    ax=plt.gca()
    ax.autoscale(enable=True)
    plt.plot(x_axis, norm.pdf(x_axis,mean1[i],var1[i]))
    plt.plot(x_axis, norm.pdf(x_axis,mean2[i],var2[i]))
    plt.show()