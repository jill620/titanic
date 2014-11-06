""" Writing my logistic regression code.
The input data are (x,y,x^2,y^2,x*y,answer) such that answer = 0 for x^2 + y^2 <= 16 
and answer = 1 for others.
The result gives coefficient
[[ -5.41963344e-06]
 [ -5.41970135e-06]
 [  2.78796859e-01]
 [  2.78796526e-01]
 [ -1.44068866e-06]]
Intercept
[-3.97983987]
Basically it's saying that the boundary is to compare whether 0.2788*(x^2 + y^2) - 4 > or < 0.
This makes sense because we know the boundary locates at x^2 + y^2 = 16, and 0.2788*16 = 4.46
""" 
import pandas as pd
import numpy as np
import csv as csv
#from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, datasets

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('temp7.txt')        # Load the train file into a dataframe
# train_df = pd.read_csv('circle_data.txt');
train_data = train_df.values
#train_data2 = np.concatenate((train_data[:,0],train_data[:,1],np.power(train_data[:,0],2),np.multiply(train_data[:,0],train_data[:,1]),train_data[:,2]),1)
#train_data3 = train_data2.values
# [:, np.newaxis] changes data format from [] to [[]], which actually has a column.
print train_data.shape
print train_data[0::,0:5].shape
#print 'Training...'
#forest = RandomForestClassifier(n_estimators=100)
#forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
# print test_data[0::,3:4].shape
LogisR = linear_model.LogisticRegression(fit_intercept=True)
LogisR.fit( train_data[0::,0:5], train_data[0::,5] )
Para = LogisR.get_params();
print "coefficient"
print LogisR.coef_.T 
print "Intercept"
print LogisR.intercept_
#print 'Predicting...'
# output = LogisR.predict(test_data[0::,0:1]).astype(int)
output = LogisR.predict(test_data)

print output

predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
