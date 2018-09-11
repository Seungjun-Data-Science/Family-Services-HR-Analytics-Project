
# coding: utf-8

# # Factor Analysis & Predicting Potential Turnover Workers

# ## Importing Libraries and Looking at Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[3]:


data= pd.read_csv('clean_data.csv')
data.head()


# In[4]:


# Dropping unncessary column
data = data.drop(labels=['Unnamed: 0'], axis=1)


# ## Preparing for Modeling

# In[5]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


# https://www.kaggle.com/randylaosat/predicting-employee-kernelover

# - https://towardsdatascience.com/predict-employee-turnover-with-python-da4975588aa3
# - https://www.kaggle.com/carinahu/employee-turnover-rate
# - https://www.kaggle.com/eeshanmishra/employee-attrition-prediction-python
# - https://www.kaggle.com/aliu233/employee-turnover-prediction
# - https://dzone.com/articles/employee-turnover-prediction-with-deep-learning
# - https://www.govloop.com/community/blog/diagnose-forecast-employee-turnover-predictive-analytics/
# - https://www.linkedin.com/pulse/analyzing-employee-turnover-predictive-methods-richard-rosenow-pmp/

# In[6]:


data.info()


# In[7]:


# Dropping unncessary columns for modeling

data_model = data.drop(labels=['Position Effective Date','Hire Date','Race','Tax ID'], axis=1)


# In[8]:


data_model['Years of Service'] = data_model.fillna(np.median(data_model['Years of Service']))
del data_model['Hire Year']
data_model['Hire Month'] = data_model.fillna(data_model['Hire Month'].value_counts().values[0])


# #### Encoding Categorical Variables

# In[9]:


X = data_model.drop('Left', axis=1) 
y = data_model['Left']


# In[10]:


encoding_needed_features = data_model.select_dtypes(include=['object']).columns.tolist()

labelEncoder = LabelEncoder()

for col in encoding_needed_features:
    X[col] = labelEncoder.fit_transform(X[col])


# #### Splitting Data into train and test set

# In[11]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)


# ## Various Predictive Classification Models

# #### Feature Importance with Decision Tree Model

# In[12]:


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

dtree = DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)

## plot the importances ##
importances = dtree.feature_importances_
feat_names = X.columns

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])


# 'Annual Salary','Age','EEOC Job Classification' and 'Regular Pay Rate Amount' are the top 4 variables that are influential in the classification process. I will use them to create a logistic regression model

# #### Logistic Regression

# In[13]:


# Move the reponse variable "turnover" to the front of the table
front = data_model['Left']
data_model.drop(labels=['Left'], axis=1,inplace = True)
data_model.insert(0, 'Left', front)

# Create an intercept term for the logistic regression equation
df = data_model.copy()
df['int'] = 1
indep_var = ['Annual Salary','Age','EEOC Job Classification','Regular Pay Rate Amount','int', 'Left']
df = df[indep_var]

target_name = 'Left'
logit_X = df.drop('Left', axis=1)
logit_y = df[target_name]

# Encoding Categorical Variables
labelEncoder = LabelEncoder()

logit_X['EEOC Job Classification'] = labelEncoder.fit_transform(logit_X['EEOC Job Classification'])

# Create train and test splits
logit_X_train, logit_X_test, logit_y_train, logit_y_test = train_test_split(logit_X,logit_y,test_size=0.2, random_state=42, stratify=y)


# scikit learn's logistc regression model already includes regularization and this is more helpful for predicting unseen data but since we are using this model on pre-existing data, I will use statsmodel.api's Logit to get the coefficients

# In[14]:


import statsmodels.api as sm
iv =  ['Annual Salary','Age','EEOC Job Classification','Regular Pay Rate Amount','int']
logReg = sm.Logit(logit_y_train, logit_X_train[iv])
answer = logReg.fit()

print(answer.summary)
print(answer.params)


# - Employee Turnover Probability = Annual Salary(-0.000011) + Age(-0.002940) + EEOC_Job_Classification(0.055412) + Regular_Pay_Rate_Amount(-0.000045) + 1.713270
# - The constant 1.713270 represents the effect of all uncontrollable variables

# In[15]:


final_df = data.copy()

for col in encoding_needed_features:
    final_df[col] = labelEncoder.fit_transform(final_df[col])


# In[16]:


final_df['Logistic Regression Quit Probability'] =final_df['Annual Salary']*(-0.000011) + final_df['Age']*(-0.002940) + final_df['EEOC Job Classification']*(0.055412) +final_df['Regular Pay Rate Amount']*(-0.000045) + 1.713270


# #### Random Forest

# In[17]:


rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    )


# In[18]:


rf.fit(X, y)


# In[19]:


final_df = final_df.join(pd.DataFrame(rf.predict_proba(X))[[1]])


# In[20]:


final_df = final_df.rename(columns={1:'Random Forest Quit Probability'})


# In[45]:


# Feature Importances with Random Forest

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90, fontsize=14)
plt.xlim([-1, X.shape[1]])


# #### Ada Boost

# In[21]:


ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
ada.fit(X,y)


# In[22]:


final_df = final_df.join(pd.DataFrame(ada.predict_proba(X))[[1]])


# In[23]:


final_df = final_df.rename(columns={1:'Ada Boost Quit Probability'})


# #### Mean Probability of leaving(quitting) of all three above models combined

# In[24]:


# Mean probability of turnover of all three models combined
final_df['Mean of Quit Probability of three models'] =(final_df['Logistic Regression Quit Probability']+final_df['Random Forest Quit Probability']+final_df['Ada Boost Quit Probability'])/3


# In[25]:


# Concatenating turnover probability to orignial data
data = data.join(final_df[['Logistic Regression Quit Probability','Random Forest Quit Probability','Ada Boost Quit Probability',
                   'Mean of Quit Probability of three models']])


# In[26]:


# Classifying each employee into different turnover risk categories (high risk, medium risk, low risk, safe)
def classify_turnover_risk(x):
    if x['Mean of Quit Probability of three models'] >= 0.9:
        return 'High Risk (> 90%)'
    elif 0.6<= x['Mean of Quit Probability of three models'] < 0.9:
        return 'Medium Risk (60~90%)'
    elif 0.2<= x['Mean of Quit Probability of three models'] < 0.6:
        return 'Low Risk (60~90%)'
    else:
        return 'Safe (< 20%)'


# In[27]:


data['Turnover Risk'] = data.apply(classify_turnover_risk, axis=1)


# In[35]:


current_employees_turnover_risk = data[data['Left']==0].reset_index()


# In[38]:


del current_employees_turnover_risk['index']


# In[39]:


current_employees_turnover_risk['Turnover Risk'].value_counts()


# In[40]:


current_employees_turnover_risk['Turnover Risk'].value_counts()/sum(current_employees_turnover_risk['Turnover Risk'].value_counts())


# In[30]:


current_employees_turnover_risk.to_csv('Current Employees_Turnover Risk.csv')

