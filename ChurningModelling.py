import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


churning_data = pd.read_csv(r'D:\PyProjects\Kaggle\Churning\Data\Data_Churn_Modelling.csv')

def printConfusionMatrix (name , y_true , y_pred):    
    cm = confusion_matrix(y_true, y_pred).ravel()
    print (name)
    print ('==============================')
    print ("True Negatives : " + str(cm[0]))
    print ("True Positives : " + str(cm[3]))
    print ("False Positives : " + str(cm[1]))
    print ("False Negatives : " + str(cm[2]))
    print ('==============================')


def getModelPerformance (y_true , y_pred):
    cm = confusion_matrix(y_true, y_pred).ravel()    
    sensitivity = cm[3] / (cm[3] + cm[2])    
    specificity = cm[0] / (cm[0] + cm[1])
    accuracy = (cm[0] + cm[3]) / (cm[0] + cm[1] + cm[2] + cm[3])
    percision = cm[3] / (cm[3] + cm[1])
    recall = cm[3] / (cm[3] + cm[2])
    MCC = ((cm[3] * cm[0]) - (cm[1]*cm[2])) / (np.sqrt((cm[3] + cm[1]) * (cm[3] + cm[2]) * (cm[0] + cm[1]) * (cm[0] + cm[2])))
    return [sensitivity , specificity , accuracy , percision , recall , MCC ]


def printPerformanceData (name , p):
    print (name)    
    print ('==============================')
    print ("Sensitivity : " + str(p[0]))
    print ("Specificity : " + str(p[1]))
    print ("Accuracy : " + str(p[2]))    
    print ("Precision : " + str(p[3]) )
    print ("Recall : " + str(p[4]) )
    print ("Mathew Accuracy : " + str(p[5]))
    print ('==============================')


def plotLR_ROC (nsProb , lr_probs , y_test , X_test ):
    ns_probs = np.array([nsProb for _ in range(len(y_test))])        
    lr_fpr, lr_tpr , lr_thr = roc_curve(y_test, lr_probs)
    ns_fpr, ns_tpr , ns_thr = roc_curve(y_test, ns_probs)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Null model')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# %% Data Summary
print(churning_data.shape)
print(churning_data.head)

# %% Data Visualization
## Visualize the two classes  (1 , 0) for exited (The dependent variable)

plt.figure(figsize=(10,7), dpi= 80)
exited_df = churning_data.groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
fig, axs = plt.subplots(figsize=(10, 7))
axs.bar(exited_df['Exited'] , exited_df['total_number'] , color="dodgerblue") 
axs.set_title('Churning')
plt.xticks(exited_df['Exited'] , ('Stays' , 'Exits'))
plt.show()

# %% Churning and Gender 
exit_females = churning_data.loc[churning_data.Gender == 'Female'].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
exit_males = churning_data.loc[churning_data.Gender == 'Male'].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()

width = 0.2

fig, axs = plt.subplots(figsize=(10, 7))
axs.bar(exit_females['Exited'] - width/2 , exit_females['total_number'] , width , color="orange" , label='Female') 
axs.bar(exit_males['Exited'] + width/2 , exit_males['total_number'] , width ,  color="dodgerblue" , label='Male') 
axs.legend()
axs.set_title('Churning by Gender')
plt.xticks(exit_females['Exited'] , ('Stays' , 'Exits'))
plt.show()

# %% Churning and Geography 
exit_france = churning_data.loc[churning_data.Geography == 'France'].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
exit_spain = churning_data.loc[churning_data.Geography == 'Spain'].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
exit_germany = churning_data.loc[churning_data.Geography == 'Germany'].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()

width = 0.3

fig, axs = plt.subplots(figsize=(10, 7))
axs.bar(exit_france['Exited'] + width/2 , exit_france['total_number'] , width , color="orange" , label='France') 
axs.bar(exit_spain['Exited'] - width/2, exit_spain['total_number'] , width ,  color="dodgerblue" , label='Spain') 
axs.bar(exit_germany['Exited'] + 1.5 * width , exit_germany['total_number'] , width ,  color="red" , label='Germany') 

axs.legend()
axs.set_title('Churning by Country')
plt.xticks(exit_females['Exited'] , ('Stays' , 'Exits'))
plt.show()

# %% Churning and Credit cards
exit_hasCrCard = churning_data.loc[churning_data.HasCrCard == 1 ].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
exit_NoCrCard = churning_data.loc[churning_data.HasCrCard == 0].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()

width = 0.3

fig, axs = plt.subplots(figsize=(10, 7))
axs.bar(exit_hasCrCard['Exited'] + width/2 , exit_hasCrCard['total_number'] , width , color="orange" , label='Has Credit Card') 
axs.bar(exit_NoCrCard['Exited'] - width/2, exit_NoCrCard['total_number'] , width ,  color="dodgerblue" , label='No Credit Card') 

axs.legend()
axs.set_title('Credit Cards')
plt.xticks(exit_females['Exited'] , ('Stays' , 'Exits'))
plt.show()

# %% Is Active member 
exit_IsActiveMember = churning_data.loc[churning_data.IsActiveMember == 1 ].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()
exit_NotActiveMember = churning_data.loc[churning_data.IsActiveMember == 0].groupby('Exited').agg(total_number = ('Exited' , 'count')).reset_index()

width = 0.3

fig, axs = plt.subplots(figsize=(10, 7))
axs.bar(exit_IsActiveMember['Exited'] + width/2 , exit_IsActiveMember['total_number'] , width , color="orange" , label='Active Member') 
axs.bar(exit_NotActiveMember['Exited'] - width/2, exit_NotActiveMember['total_number'] , width ,  color="dodgerblue" , label='Not active Member') 

axs.legend()
axs.set_title('Active Member')
plt.xticks(exit_females['Exited'] , ('Stays' , 'Exits'))
plt.show()

# %% Credit Score 
stays = churning_data.loc[churning_data.Exited == 1, 'CreditScore']
exits = churning_data.loc[churning_data.Exited == 0, 'CreditScore']

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title("Credit Score")
sns.distplot(exits, color="orange", label="Exits", **kwargs)
plt.legend();


# %% Visualize distribution Salary.
stays = churning_data.loc[churning_data.Exited == 1, 'EstimatedSalary']
exits = churning_data.loc[churning_data.Exited == 0, 'EstimatedSalary']

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title("Churning and Salary")
sns.distplot(exits, color="orange", label="Exits", **kwargs)
plt.legend();
# %% Distribution of Numebr of products
stays = churning_data.loc[churning_data.Exited == 1, 'NumOfProducts']
exits = churning_data.loc[churning_data.Exited == 0, 'NumOfProducts']

kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title("Number of Products")
sns.distplot(exits, color="orange", label="Exits", **kwargs)

plt.legend();

# %% Age distribution 
stays = churning_data.loc[churning_data.Exited == 1, 'Age']
exits = churning_data.loc[churning_data.Exited == 0, 'Age']

kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title('Churning by Age')
sns.distplot(exits, color="orange", label="Exits", **kwargs)

plt.legend();

# %% Balance distribution
stays = churning_data.loc[churning_data.Exited == 1, 'Balance']
exits = churning_data.loc[churning_data.Exited == 0, 'Balance']

kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title('Balance')
sns.distplot(exits, color="orange", label="Exits", **kwargs)

plt.legend();

# %% Tenure distributions
stays = churning_data.loc[churning_data.Exited == 1, 'Tenure']
exits = churning_data.loc[churning_data.Exited == 0, 'Tenure']

kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2})

plt.figure(figsize=(10,7), dpi= 80)

sns.distplot(stays, color="dodgerblue", label="Stays", **kwargs).set_title('Tenure')
sns.distplot(exits, color="orange", label="Exits", **kwargs)

plt.legend();

# %% Data split to train and test datasets.

## data info. 
label_encoder = LabelEncoder()
## Categorical data encoding , Gender , Geography. 
churning_data['Gender'] = label_encoder.fit_transform(churning_data['Gender']) ## Gender (Female , Male)
churning_data['Geography'] = label_encoder.fit_transform(churning_data['Geography']) ## Geography (France , Spain ...etc)

## predictors : Credit Score , Geography , Gender , Age , Tenure , Balance , NumOfProducts , HasCrCard , IsActiveMember , EstimatedSalary
X = churning_data[['CreditScore' , 'Geography' , 'Gender' , 'Age' , 'Tenure' , 'Balance' , 'NumOfProducts' , 'HasCrCard' , 'IsActiveMember' , 'EstimatedSalary']]

## dependent variable , "Exited"
y = churning_data[['Exited']]

## Split data into training and test sets. Use following proportions  train 70% and test 30%
X_train, X_test, y_train, y_test =  train_test_split (X , y , test_size=0.3 , random_state = 42)

# %% Null model , no predictors. 
positives = churning_data[churning_data['Exited'] == 1]
positivePercent = len(positives) / len(churning_data)
print("Positive (exited = 1) in Null model  : " + str(positivePercent))
print("Negatives (exited = 0) in Null model  : " + str(1 - positivePercent))

# %% Logistic regression without regularization.
lr1 = LogisticRegression()
lr1.fit(X_train , y_train.values.ravel())

y_test_pred1 = lr1.predict(X_test)
printConfusionMatrix ("Test Data Confusion Matrix" , y_test , y_test_pred1)
performanceTest = getModelPerformance (y_test , y_test_pred1)
printPerformanceData("Logistic Regression Test Data Performance" , performanceTest)
lr_probs = lr1.predict_proba(X_test)[:, 1]
plotLR_ROC((1-positivePercent) , lr_probs , y_test , X_test)

# %% logistic regression with regularization.
lr2 = LogisticRegression( penalty='l2' , C=1 , class_weight='balanced' , solver='liblinear')
lr2.fit(X_train , y_train.values.ravel())

lr_probs = lr2.predict_proba(X_test)[:, 1]
y_test_pred2 = lr_probs > 0.55
printConfusionMatrix ("Confusion Matrix for Logistic Regression balanced with penality" , y_test , y_test_pred2)
performanceTest2 = getModelPerformance (y_test , y_test_pred2)
printPerformanceData("Test Data Performance" , performanceTest2)
plotLR_ROC((1-positivePercent) , lr_probs , y_test , X_test)

# %% Classification with Tree model

## Use of cross validation to find the best depth of decision tree.
param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 10)}    
ct = tree.DecisionTreeClassifier()   
ct_gscv = GridSearchCV(ct, param_grid, cv=20)    
ct_gscv.fit(X_train, y_train)
print ("tuned hpyerparameters :(best parameters) " + str(ct_gscv.best_params_))

ct_probs = ct_gscv.predict_proba(X_test)[:, 1]
y_test_pred4 = ct_probs > 0.30
printConfusionMatrix ("Classification Tree Test Data Confusion Matrix" , y_test , y_test_pred4)
performanceTest4 = getModelPerformance (y_test , y_test_pred4)
printPerformanceData("Test Data Performance" , performanceTest4)

plotLR_ROC((1-positivePercent) , ct_probs , y_test , X_test)

# %% Random Forest 
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train);

y_test_pred5 = rf.predict(X_test) >= 0.3
printConfusionMatrix ("Classification Tree Test Data Confusion Matrix" , y_test , y_test_pred5)
performanceTest5 = getModelPerformance (y_test , y_test_pred5)
printPerformanceData("Test Data Performance" , performanceTest5)
rf_prob = rf.predict(X_test)
plotLR_ROC((1-positivePercent) , rf_prob , y_test , X_test)

# %%
