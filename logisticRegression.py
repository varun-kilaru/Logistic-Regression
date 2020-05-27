import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')

#reading the data
churn = pd.read_csv('churn_data.csv')
customer = pd.read_csv('customer_data.csv')
internet = pd.read_csv('internet_data.csv')
# print(churn.head())
# print(customer.head())
# print(internet.head())

#combing the data to create a single data frame
df_1 = pd.merge(churn, customer, how='inner', on='customerID')
df = pd.merge(df_1, internet, how='inner', on='customerID')

#info about data
print('\nShape :')
print('###################################################')
print(df.shape)
print('###################################################')

print('\nHead :')
print('###################################################')
print(df.head())
print('###################################################')

print('\nDescribe :')
print('###################################################')
print(df.describe())
print('###################################################')

print('\nInfo :')
print('###################################################')
print(df.info())
print('###################################################')

#encoding the binary categorical variables(yes/no)
bin_vars = ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']

#function to encode binary variables to 1 or 0
def binary_map(x):
    return x.map({'Yes':1, 'No':0})

#encoding those binary variables
df[bin_vars] = df[bin_vars].apply(binary_map)
print('\nData :\n',df.head())

#encoding multi level categorical variables
cat_vars = ['Contract','PaymentMethod','gender','MultipleLines','InternetService',\
    'OnlineSecurity','TechSupport','StreamingTV','OnlineBackup',\
        'DeviceProtection','StreamingMovies']
''' Using get_dummies() would give no.of columns that is
equal to no.of levels in that categorical variable.
For eg, we have a size column with three levels - large, medium, small.
Using get_dummies() would return three columns as
size        size_large    size_medium       size_small
large            1             0                0
medium           0             1                0
small            0             0                0'''
''' Here we choose the first column to be droppped for optimization.'''
dummy1 = pd.get_dummies(df[['Contract', 'PaymentMethod', 'gender', 'InternetService']]
    , drop_first=True)
df = pd.concat([df, dummy1], axis=1)
# print(df.head())
''' Here we choose the droppped columns manually that are
not useful.'''
ml = pd.get_dummies(df['MultipleLines'], prefix='MultipleLines')
ml1 = ml.drop(['MultipleLines_No phone service'], 1)
df = pd.concat([df, ml1], axis=1)

os = pd.get_dummies(df['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], 1)
df = pd.concat([df, os1], axis=1)

ob = pd.get_dummies(df['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], 1)
df = pd.concat([df, ob1], axis=1)

dp = pd.get_dummies(df['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], 1)
df = pd.concat([df, dp1], axis=1)

ts = pd.get_dummies(df['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], 1)
df = pd.concat([df, ts1], axis=1)

st = pd.get_dummies(df['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], 1)
df = pd.concat([df, st1], axis=1)

sm = pd.get_dummies(df['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], 1)
df = pd.concat([df, sm1], axis=1)

#dropping the categorical variables after encoding them
df = df.drop(df[cat_vars], 1)

#changing datatype of TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# print(df.info())

#checking for outliers & linearity
num_vars = ['tenure', 'MonthlyCharges', 'SeniorCitizen', 'TotalCharges']
print('\nLinearity Check :\n')
print(df[num_vars].describe(percentiles=[.25, .5, .75, .9, .95, .99 ]))

#checking for missing values
print('\nMissing Values :\n')
print(pd.isnull(df).sum())
print('\nMissing Values % :')
print(round(100*(df.isnull().sum())/len(df.index), 2))

''' As the missing values percentage is 0.16
we could remove those missing value rows '''
#removing missing value rows & checking them
df = df[~np.isnan(df['TotalCharges'])]
print('\nAgain checking Missing Values :\n')
print(pd.isnull(df).sum())

#creatinhg train & test data
X = df.drop(['Churn', 'customerID'], axis=1)
# print(X.head())
y = df['Churn']
# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

#feature scaling using standardization
scaler = StandardScaler()
X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])
# print(X_train.head())

#checking Churn rate(i.e class imbalancing)
churn_rate = (sum(df['Churn'])/len(df['Churn'].index))*100
print('\nchurn rate : ',churn_rate)

#checking the correlations
plt.figure(figsize=(20, 16))
plt.title('Correlations in data')
sns.heatmap(df.corr(), annot=True)
plt.show()

#dropping correlated features manually
X_train = X_train.drop(['MultipleLines_No', 'OnlineSecurity_No', 'OnlineBackup_No',\
    'DeviceProtection_No', 'TechSupport_No', 'StreamingTV_No', 'StreamingMovies_No'], 1)
# print(X_train.head())

X_test = X_test.drop(['MultipleLines_No', 'OnlineSecurity_No', 'OnlineBackup_No',\
    'DeviceProtection_No', 'TechSupport_No', 'StreamingTV_No', 'StreamingMovies_No'], 1)

#once again checking correlations
plt.figure(figsize=(20, 16))
plt.title('Correlations in train data')
sns.heatmap(X_train.corr(), annot=True)
plt.show()

#building model using sklearn for feature selection
from sklearn.linear_model import LogisticRegression

lgre = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(lgre, 15)
rfe.fit(X_train, y_train)

#creating a predictors table to choose predictors(i.e, the result of RFE)
rfe_df = pd.DataFrame()
rfe_df['Features'] = X_train.columns
rfe_df['Select/Not'] = rfe.support_
rfe_df['Ranking'] = rfe.ranking_
print('\nRFE result :\n')
print(rfe_df)

selected_features = X_train.columns[rfe.support_]
# print(selected_features)
''' Here the RFE selects the following predictors:
['tenure', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'SeniorCitizen', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check',
    'InternetService_Fiber optic', 'InternetService_No',
    'MultipleLines_Yes', 'TechSupport_Yes', 'StreamingTV_Yes',
    'StreamingMovies_Yes']
But by building a model and checking the Variance Inflation Factor(VIF)
& p-values of selected predictors,we came to know that "MonthlyCharges, TotalCharges"
have the high VIF. So, we need to eliminate "MonthlyCharges, TotalCharges" predictors
which are insigificant. '''
X_train = X_train[selected_features]
X_train = X_train.drop(['MonthlyCharges', 'TotalCharges'], axis=1)

#building model using statsmodels
import statsmodels.api as sm

#assessing model with selected features
X_train_sm = sm.add_constant(X_train)
loglm1 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
model = loglm1.fit()
print('\nModel Summary :\n',model.summary())

#Creating a Variance Inflation Factor(vif) table for the predictors
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i)for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 3)
vif = vif.sort_values(by='VIF', ascending=False)
print('\nVIF Table:\n',vif)

y_train_pred = model.predict(X_train_sm)
y_train_pred = y_train_pred.values.reshape(-1)

predictA_df = pd.DataFrame({'Actual': y_train.values, 'Probability': y_train_pred})
predictA_df['Predicted'] = predictA_df['Probability'].map(lambda x: 1 if x>0.3 else 0)
print('\nPrediction Table :\n',predictA_df)

#evaluating model
from sklearn import metrics

''' confusion matrix
___________________________________________
|    Actual       |       Predicted       |
|_________________|_______________________|
|not-churn        |   not-churn | churn   |
|  2791+844       |      2791  |  844     |
|-----------------|-----------------------|
|churn            |   not-churn | churn   |
|   288+999       |      288    |  999    |
|_________________|_______________________|'''

confusion_matrixA = metrics.confusion_matrix(predictA_df.Actual, predictA_df.Predicted)
print('Confusion Matrix : \n',confusion_matrixA)

#accuracy
acc = metrics.accuracy_score(predictA_df.Actual, predictA_df.Predicted)
print('Accuracy : ',acc)
#specificity
specificity = confusion_matrixA[0,0]/(confusion_matrixA[0,0]+confusion_matrixA[0,1])
print('Specificity : ',specificity)
#sensitivity
sensitivity = confusion_matrixA[1,1]/(confusion_matrixA[1,0]+confusion_matrixA[1,1])
print('Sensitivity : ',sensitivity)
#True Positive Rate
tpr = confusion_matrixA[1,1]/(confusion_matrixA[1,0]+confusion_matrixA[1,1])
print('True Positive Rate : ',tpr)

''' Here, we need to choose the threshold value as the
value for which the accuracy, sensitivity and specificity
are almost equal.'''
'''finding the cut off to predict yes/no based on probabilities(i.e,
reason for choosing the probability 0.3 as cut_off)'''
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    pred_labels = predictA_df['Probability'].map(lambda x: 1 if x>i else 0)
    cm1 = metrics.confusion_matrix(predictA_df.Actual, pred_labels)
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print('\nTable to choose threshold :\n',cutoff_df)

#predicting the test data
X_test = X_test[selected_features]
X_test = X_test.drop(['MonthlyCharges', 'TotalCharges'], axis=1)

X_test_sm = sm.add_constant(X_test)
y_test_pred = model.predict(X_test_sm)
y_test_pred = y_test_pred.values.reshape(-1)

predict_df = pd.DataFrame({'Actual': y_test.values, 'Probability': y_test_pred})
predict_df['Predicted'] = predict_df['Probability'].map(lambda x: 1 if x>0.3 else 0)
print('\nPrediction Table :\n',predict_df)

#evaluating model
from sklearn import metrics

''' confusion matrix
___________________________________________
|    Actual       |       Predicted       |
|_________________|_______________________|
|not-churn        |   not-churn | churn   |
|  2791+844       |      2791  |  844     |
|-----------------|-----------------------|
|churn            |   not-churn | churn   |
|   288+999       |      288    |  999    |
|_________________|_______________________|'''

confusion_matrix = metrics.confusion_matrix(predict_df.Actual, predict_df.Predicted)
print('Confusion Matrix : \n',confusion_matrix)

#accuracy
acc = metrics.accuracy_score(predict_df.Actual, predict_df.Predicted)
print('Accuracy : ',acc)
#specificity
specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Specificity : ',specificity)
#sensitivity
sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Sensitivity : ',sensitivity)
#True Positive Rate
tpr = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('True Positive Rate : ',tpr)
