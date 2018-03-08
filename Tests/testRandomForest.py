#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
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
    return(names,numpy.array(X),numpy.array(y))

def validation(clf,X,y):
    #print("Prediction:")
    pred=clf.predict(X)
    confusionMatrix=confusion_matrix(pred,y)
    #print(confusionMatrix)
    precision=sum(numpy.diagonal(confusionMatrix))/sum(sum(confusionMatrix))
    return precision
    


(names1,X1,y1)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\data1et2.csv")
(names2,X2,y2)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\test.csv")
(names3,X3,y3)=importData("D:\\Documents\\Centrale\\Ei2\\PGROU\\data.csv")


p1=[]
p2=[]



'''
clf1=GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=30, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
clf1.fit(X1, y1)
clf2=GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=30, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
clf2.fit(X2, y2)
'''

r=[]
for i in range (1000):
    seed = random.randint(1,10000)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=test_size, random_state=seed)
    #print(X_test)
    # fit model no training data
    model = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    r.append(accuracy)
    

print("Accuracy: %.2f%%" % (sum(r)/1000 * 100.0))

y_pred = model.predict(X2)
#predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y2, y_pred)


#print("Accuracy: %.2f%%" % (accuracy * 100.0))


y_pred = model.predict(X3)
#predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y3, y_pred)


#print("Accuracy: %.2f%%" % (accuracy * 100.0))






seed = 4


X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier(max_depth=2, learning_rate=0.01, n_estimators=40, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))




    
    
'''    
clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(X1,y1)
clf1 = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf1.fit(X1,y1)
'''

'''
p1.append(validation(clf1,X2,y2))
p2.append(validation(clf2,X1,y1))



plt.plot(p1)
plt.plot(p2)
print(numpy.mean(p1))
print(max(p1))
print(min(p1))
print("\n")
print(numpy.mean(p2))
print(max(p2))
print(min(p2))
    
#plt.show()
'''

