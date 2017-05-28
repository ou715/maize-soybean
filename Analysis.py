import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import matplotlib

matplotlib.style.use('ggplot')

maize = pd.read_csv('maize.csv')
soy = pd.read_csv('soybean.csv')

maizeCorr = maize.corr() 
soyCorr = soy.corr()
maizeYCorr = maizeCorr['Yield']
soyYCorr = soyCorr['Yield']

prinosM = np.array(maize['Yield'])
prinosS = np.array(soy['Yield'])
    
dataScM = preprocessing.scale(maize.iloc[:,0:-1])

trainM, testM, trainPM, testPM = train_test_split(dataScM, prinosM, test_size=0.15)
ridgeM = linear_model.Ridge(alpha=11.6)
ridgeM.fit(trainM, trainPM)

ridgePM = ridgeM.predict(testM)
skorM = cross_val_score(ridgeM, dataScM ,prinosM, cv=10, scoring ='neg_median_absolute_error')
skorM = -1*skorM
red = range(len(testPM))+np.ones(len(testPM))
plt.figure(0)
plt.scatter(red,testPM, alpha = 0.8)
plt.scatter(red,ridgePM, marker = 'P', alpha = 0.8)

plt.legend(('Actual','Prediction \n $R^2$ = %0.3f' % ridgeM.score(testM,testPM)), fontsize = 14)
plt.title('Maize yield' , fontsize = 20)
plt.savefig('Maize yield', dpi = 450)
print("Accuracy for maize (R^2): %0.2f (+/- %0.2f)" % (np.mean(skorM), skorM.std() * 2))
maizeA = list()
for i in range(100):
    ridgeM = linear_model.Ridge(alpha=i*0.1)
    ridgeM.fit(trainM, trainPM)
    maizeA.append(np.mean(cross_val_score(ridgeM,dataScM, prinosM, cv =10,scoring ='neg_median_absolute_error')))


    
dataScS = preprocessing.scale(soy.iloc[:,0:-1])

trainS, testS, trainPS, testPS = train_test_split(dataScS, prinosS, test_size=0.15)
ridgeS = linear_model.Ridge(alpha=2.4)
ridgeS.fit(trainS, trainPS)

ridgePS = ridgeS.predict(testS)
skorS = cross_val_score(ridgeS, dataScS ,prinosS, cv=10,scoring ='neg_median_absolute_error')
skorS = -1 * skorS
red = range(len(testPS))+np.ones(len(testPS))
plt.figure(1)
plt.scatter(red,testPS, alpha = 0.8)
plt.scatter(red,ridgePS, marker = 'P', alpha = 0.8)

plt.legend(('Actual','Prediction \n $R^2$ = %0.3f' % ridgeS.score(testS,testPS)), fontsize = 14)
plt.title('Soybean yield' , fontsize = 20)
plt.savefig('Soy prediction',dpi=450)
print("Accuracy for soybean (R^2): %0.2f (+/- %0.2f)" % (np.mean(skorS), skorS.std() * 2))
soyA = list()
for i in range(100):
    ridgeS = linear_model.Ridge(alpha=i*0.1)
    ridgeS.fit(trainS, trainPS)
    soyA.append(np.mean(cross_val_score(ridgeS,dataScS, prinosS, cv =10,scoring ='neg_median_absolute_error')))


predicted = cross_val_predict(ridgeM, dataScM, prinosM, cv=10)
plt.figure(2)
fig, ax = plt.subplots()
plt.title('Maize yield', fontsize = 20)
ax.scatter(prinosM, predicted, color = 'y')
plt.legend(('MAD = %0.3f' % np.mean(skorM),), fontsize = 14)
ax.plot([prinosM.min(), prinosM.max()], [prinosM.min(), prinosM.max()], 'k--', lw=4)
ax.set_xlabel('Measured', fontsize =14)
ax.set_ylabel('Predicted', fontsize = 14)
plt.savefig('Maize Yield', dpi=450)
plt.show()
    
predicted = cross_val_predict(ridgeS, dataScS, prinosS, cv=10)
plt.figure(3)
fig, ax = plt.subplots()
plt.title('Soybean yield', fontsize =20)

ax.scatter(prinosS, predicted, color = 'g')
plt.legend(('MAD = %0.3f' % np.mean(skorS),), fontsize = 14)
ax.plot([prinosS.min(), prinosS.max()], [prinosS.min(), prinosS.max()], 'k--', lw=4)
ax.set_xlabel('Measured', fontsize = 14)
ax.set_ylabel('Predicted', fontsize = 14)

plt.savefig('Soybean Yield', dpi=450)
plt.show()

print('Done!')