
# coding: utf-8

# # Data Cleaning + Wrangling

# ## Importing Libraries, Merging data sets, Handling duplicates and weird labels

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_excel('hr_data_final.xlsx')


# In[3]:


data2 = pd.read_excel('curr_employee.xls')


# In[4]:


data3 = pd.read_excel('turnover.xls')


# In[5]:


data4 = pd.read_excel('degree_job_class_filled.xls')


# In[6]:


# Dropping Duplicate Columns from data
data = data.loc[:,~data.columns.duplicated()]


# In[7]:


# Dropping duplicates and keeping the first ones
data = data.drop_duplicates(subset=['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])


# In[8]:


# Dropping duplicates and keeping the first ones
data2 = data2.drop_duplicates(subset=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])


# In[9]:


# Dropping duplicates and keeping the first ones
data3 = data3.drop_duplicates(subset=['Gender','Annual Salary','Age','Position Effective Date'])


# In[10]:


fin_data = pd.merge(data3, data, on=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'],how='outer')


# In[11]:


# Dropping duplicates and keeping the first ones
fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date'])


# In[12]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+"_y"]


# In[13]:


for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():
    del fin_data[col]


# In[14]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data = fin_data.rename(columns={col:col[:-2]})


# In[15]:


# Deleting unnecessary columns in data
for col in ['Education Level Description.1','Annual Salary.1','Pay Frequency.1','Regular Pay Rate Amount.1','Education Level Description']:
    del fin_data[col]


# In[16]:


# Dropping duplicates and keeping the first ones
fin_data = fin_data.drop_duplicates(subset=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])


# In[17]:


fin_data = pd.merge(fin_data, data2,on=['Age','Annual Salary','Ethnicity','Gender','Position Effective Date'], how='left')


# In[18]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+"_y"]


# In[19]:


for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():
    del fin_data[col]


# In[20]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data = fin_data.rename(columns={col:col[:-2]})


# In[21]:


del fin_data['Unnamed: 6']


# In[22]:


del fin_data['Business Unit Description']


# In[23]:


# Dropping duplicates and keeping the first ones
fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date'])


# In[24]:


data4['Education Level Code'] = data4['Education Level Description']


# In[25]:


del data4['Education Level Description']


# In[26]:


fin_data = pd.merge(fin_data,data4[['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date',
                                    'Education Level Code','EEOC Job Classification']], 
                    on=['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date'], how='left')


# In[27]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+"_y"]


# In[28]:


for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():
    del fin_data[col]


# In[29]:


for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():
    fin_data = fin_data.rename(columns={col:col[:-2]})


# In[30]:


fin_data.info()


# In[31]:


# Dropping duplicates and keeping the first ones
fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date','Years of Service'])


# In[32]:


fin_data.info()


# ## Missing Data

# In[33]:


fin_data['Age'] = fin_data['Age'].fillna(np.median(fin_data['Age']))
fin_data['Ethnicity'] = fin_data['Ethnicity'].fillna('Hispanic or Latino')
fin_data['Race'] = fin_data['Race'].fillna('Hispanic or Latino')


# In[34]:


# If two employees have same job classification, assumed they work same number of hours
fin_data['Scheduled Hours'] = fin_data.groupby('EEOC Job Classification')['Scheduled Hours'].bfill().ffill()


# In[35]:


fin_data['Scheduled Hours'].value_counts().index[0]


# In[36]:


# new race code 10.0 denoting "Hispanic or Lationo"
fin_data['Race Code'] = fin_data['Race Code'].fillna(10.0)


# In[37]:


# If two employees have same home department, Job Title Description and number of work hours, assumed their regular pay amount is equal
fin_data['Regular Pay Rate Amount'] =fin_data.groupby(['EEOC Job Classification','Scheduled Hours','Home Department Description'])['Regular Pay Rate Amount'].bfill().ffill()


# In[38]:


# Fill in the still missing values with the most frequent value of that variable
fin_data['Scheduled Hours'] = fin_data['Scheduled Hours'].fillna(fin_data['Scheduled Hours'].value_counts().index[0])
fin_data['Regular Pay Rate Amount'] = fin_data['Regular Pay Rate Amount'].fillna(fin_data['Regular Pay Rate Amount'].value_counts().index[0])


# In[39]:


del fin_data['Position Start Date']


# In[40]:


del fin_data['Promotion Check']


# In[41]:


del fin_data['Home Department Code']


# In[42]:


# If position effective data doesn't match hiring date, means the employee has been promoted
fin_data['check'] = np.where(fin_data['Position Effective Date']!=fin_data['Hire Date'], 'Yes', 'No')


# In[43]:


# If position effective data doesn't match hiring date, means the employee has been promoted
fin_data.loc[fin_data['Promotion'].isnull(),'Promotion'] =fin_data['check']


# In[44]:


del fin_data['check']


# In[45]:


del fin_data['Race Description']


# In[46]:


# If two employees have same home department, Job Classification and number of work hours, assumed their Supervisor is the same
fin_data['Supervisor ID'] =fin_data.groupby(['EEOC Job Classification','Home Department Description','Scheduled Hours'])['Supervisor ID'].bfill().ffill()


# In[47]:


fin_data['Supervisor Name'] = fin_data['Reports To First Name'] + " " + fin_data['Reports To Last Name']


# In[48]:


# If two employees have the smae supervisor ID, then the supervisor's name would be the same
fin_data['Supervisor Name'] =fin_data.groupby(['Supervisor ID'])['Supervisor Name'].bfill().ffill()


# In[49]:


# If two employees have same anuual salary, scheduled number of hours and regular pay amount, 
# assumed their job classification would be the same
fin_data['EEOC Job Classification'] =fin_data.groupby(['Annual Salary','Scheduled Hours','Regular Pay Rate Amount'])['EEOC Job Classification'].bfill().ffill()


# In[50]:


del fin_data['Current Date']


# In[51]:


del fin_data['Job Change Reason Code']


# In[52]:


del fin_data['Reports To First Name']
del fin_data['Reports To Last Name']


# In[53]:


del fin_data['Benefits Eligibility Class Code']


# In[54]:


# If two employees have the same Job Classification, number of work hours, regular pay amount, and annual salary,
# assumed their job titles would be the same
fin_data['Job Title Description'] =fin_data.groupby(['EEOC Job Classification','Scheduled Hours','Regular Pay Rate Amount','Annual Salary'])['Job Title Description'].bfill().ffill()


# In[55]:


del fin_data['Pay Frequency']


# In[56]:


# If two employees have the same Job Classification, job description, regular pay amount, and annual salary, 
# assumed FLSA Code would be the same
fin_data['FLSA Code'] =fin_data.groupby(['EEOC Job Classification','Job Title Description','Regular Pay Rate Amount','Annual Salary'])['FLSA Code'].bfill().ffill()


# In[57]:


fin_data['FLSA Code'] = fin_data['FLSA Code'].replace({' ':np.nan})


# In[58]:


# If two employees have the same Job Classification, job description, regular pay amount, and annual salary, 
# assumed FLSA Code would be the same
fin_data['FLSA Code'] =fin_data.groupby(['EEOC Job Classification','Job Title Description','Regular Pay Rate Amount','Annual Salary'])['FLSA Code'].bfill().ffill()


# In[59]:


# If two employees have the same Annual Salary, job description, and regular pay amount, assumed Education Level would be the similar(same)
fin_data['Education Level Code'] =fin_data.groupby(['Job Title Description','Regular Pay Rate Amount','Annual Salary'])['Education Level Code'].bfill().ffill()


# In[60]:


# If two employees have the same supervisor, job classification and job description, assumed Home Department would be the same
fin_data['Home Department Description'] =fin_data.groupby(['Supervisor ID','EEOC Job Classification','Job Title Description'])['Home Department Description'].bfill().ffill()


# In[61]:


# Column that indicates whether this employee is still in Family Services or has been terminated(left)
fin_data['Left'] = np.where(fin_data['Termination Date'].notnull(), 1, 0)


# In[62]:


# Fill null with today's timestamp
from pandas import Timestamp
fin_data['Termination Date'] = fin_data['Termination Date'].fillna(Timestamp.today())


# In[63]:


# Years of service in timedelta
fin_data['timedelta'] = fin_data['Termination Date'] - fin_data['Hire Date']


# In[64]:


# Converting Years of service in timedelta into number of years
fin_data['timedelta'] = (fin_data['timedelta'].dt.days)/365


# In[65]:


fin_data.loc[fin_data['Years of Service'].isnull(),'Years of Service']=fin_data['timedelta']


# In[66]:


del fin_data['timedelta']


# In[67]:


del fin_data['Termination Date']


# In[68]:


# If two employees have the same job classification, job description, and annual salary assumed Benefits Eligibility would be the same
fin_data['Benefits Eligibility Class Description'] =fin_data.groupby(['EEOC Job Classification','Job Title Description','Annual Salary'])['Benefits Eligibility Class Description'].bfill().ffill()


# In[69]:


del fin_data['Years in Current Position']


# In[70]:


# Filling in missing information of categorical variables with "Not Reported"
for col in ['Supervisor ID','Supervisor Name','FLSA Code','Home Department Description','Job Title Description',
           'Payroll Company Code','Education Level Code','Education Level Code',
            'Benefits Eligibility Class Description']:
    fin_data[col] = fin_data[col].fillna('Not Reported')


# In[71]:


fin_data['EEOC Job Classification'] = fin_data['EEOC Job Classification'].replace({'Not reported': 'Not Reported'})


# In[72]:


fin_data['FLSA Code'] = fin_data['FLSA Code'].replace({' ': 'Not Reported'})


# In[73]:


fin_data = fin_data.reset_index()


# In[74]:


# Dropping duplicates and keeping the first ones
data = data.drop_duplicates(subset=['Age','Race','Ethnicity','Annual Salary','Scheduled Hours','Regular Pay Rate Amount'])


# In[75]:


del fin_data['index']


# In[76]:


# Changing datatype of Hire Date into "datetime" for convenience
fin_data['Hire Date'] = pd.to_datetime(fin_data['Hire Date'])


# In[77]:


fin_data['Hire Year'] = data['Hire Date'].dt.year


# In[82]:


fin_data['Hire Year'] = fin_data['Hire Year'].astype('int64',errors='ignore')


# In[79]:


fin_data['Hire Month'] = fin_data['Hire Date'].dt.month


# In[85]:


fin_data['Hire Month'] = fin_data['Hire Month'].astype('int64',errors='ignore')


# In[86]:


fin_data.info()


# ## Saving into CSV Files

# In[87]:


# Save cleaned data as a new file

fin_data.to_csv('clean_data.csv')


# In[88]:


# Current Employees
curr_emp = fin_data[fin_data['Left']==0]


# In[89]:


curr_emp.to_csv('curr_emp.csv')


# In[90]:


# Terminated Employees
left_emp = fin_data[fin_data['Left']==1]


# In[91]:


left_emp.to_csv('left_emp.csv')


# In[92]:


print(fin_data.shape, curr_emp.shape, left_emp.shape)


# 652 current employees and 2652 terminated employees
