 

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import string
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.T1_V5.unique()

#Plotting Correlation graph
corr=train.corr()
plt.figure(figsize=(20,14))
sns.heatmap(corr,annot=True)

train.Hazard.value_counts()
plt.hist(train['Hazard'],bins=100,)

train=train.drop("Id",axis=1)

num_var=train.select_dtypes(include=[np.number])
num_var_test=test.select_dtypes(include=[np.number])
cat_var=train.select_dtypes(exclude=[np.number])
cat_var_test=test.select_dtypes(exclude=[np.number])

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
cat_var=cat_var.apply(LabelEncoder().fit_transform)
enc=LabelEncoder()
cat_var_test=cat_var.apply(LabelEncoder().fit_transform)


X=train.drop('Hazard',axis=1)
Y=train.Hazard

new_train=pd.concat([num_var,cat_var],axis=1)
new_test=pd.concat([num_var_test,cat_var_test],axis=1)
 
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X_train=new_train.iloc[:,1:33]
Y_train=new_train.iloc[:,0]
X_test=new_test.iloc[:,1:33]

model = RandomForestRegressor()
# fit the model
model.fit(X_train, Y_train)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

plt.figure(figsize=[10,7])
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.sort_values().plot(kind='barh')
plt.show()


import scipy.stats as stats
def chisquare(col):
    dataset_table=pd.crosstab(train[col],train['Hazard'])
    Observed_values=dataset_table.values
    val=stats.chi2_contingency(dataset_table)
    Expected_values=val[3]
    no_of_rows=dataset_table.shape[0]
    no_of_cols=dataset_table.shape[1]
    
    dof=(no_of_rows-1)*(no_of_cols-1)
    print(dof)
    alpha=0.05
    
    from scipy.stats import chi2
    chi_square=sum([(O-E)**2./E for O,E in zip(Observed_values,Expected_values)])
    chi_square_stats=chi_square[0]+chi_square[1]
    p_value=1-chi2.cdf(x=chi_square_stats,df=dof)
    print(p_value)

 

X_train=X_train.drop(['T1_V10','T1_V13','T2_V7'],axis=1)
X_test=X_test.drop(['T1_V10','T1_V13','T2_V7'],axis=1)


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, Y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

d={"Id":test.iloc[:,0].values,"Hazard":y_pred}
df=pd.DataFrame(data=d)
 

df.to_csv('lmg.csv',index=False)

from sklearn.model_selection import train_test_split
train_set, valid_set, train_labels, valid_labels = train_test_split(
    X_train, Y_train, test_size=0.4, random_state=4327)

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(train_set, train_labels)
print(accuracy_score(valid_labels, xgb_classifier.predict(valid_set)))
