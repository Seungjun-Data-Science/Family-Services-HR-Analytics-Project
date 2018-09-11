
# coding: utf-8

# # General Analysis 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Making different Clusters of employees

# In[168]:


data = pd.read_csv('clean_data.csv')


# In[3]:


del data['Unnamed: 0']


# In[4]:


curr_emp = pd.read_csv('curr_emp.csv')
quit_emp = pd.read_csv('left_emp.csv')


# In[5]:


del curr_emp['Unnamed: 0']
del quit_emp['Unnamed: 0']


# In[6]:


# Employees who quit their jobs within 1 year
quit_less_one_year_df = quit_emp[quit_emp['Years of Service'] <= 1]


# In[7]:


# Employees who quit their jobs within 2 years
quit_less_two_year_df = quit_emp[quit_emp['Years of Service'] <= 2]


# In[8]:


# Employees who quit their jobs within 3 years
quit_less_three_year_df = quit_emp[quit_emp['Years of Service'] <= 3]


# In[9]:


# Employees (both terminated and current) who worked for more than 8 or mroe years
worked_8years_df = data[data['Years of Service']>=8]


# In[10]:


# Employees (both terminated and current) who worked for more than 5 or more years
worked_5years_df = data[data['Years of Service']>=5]


# ## Comparison by various features

# #### Annual Salary

# In[11]:


f, (ax1, ax2) = plt.subplots(2)
sns.distplot(curr_emp['Annual Salary'], ax=ax1)
sns.distplot(quit_less_one_year_df['Annual Salary'],ax=ax2)
plt.tight_layout()

print("Mean Annual Salary of Current Employees :", curr_emp['Annual Salary'].mean())
print("Mean Annual Salary of Employees who quit within a year :", quit_less_one_year_df['Annual Salary'].mean())


# In[12]:


f, (ax1, ax2) = plt.subplots(2)
sns.distplot(quit_less_two_year_df['Annual Salary'],ax=ax1)
sns.distplot(quit_less_three_year_df['Annual Salary'],ax=ax2)
plt.tight_layout()

print("Employees who quit within two years: ", quit_less_two_year_df['Annual Salary'].mean())
print("Employees who quit within three years: ", quit_less_three_year_df['Annual Salary'].mean())


# In[13]:


f, (ax1, ax2) = plt.subplots(2)
sns.distplot(worked_8years_df['Annual Salary'],ax=ax1)
sns.distplot(worked_5years_df['Annual Salary'],ax=ax2)
plt.tight_layout()

print("Employees who worked for 8 or more years: ", worked_8years_df['Annual Salary'].mean())
print("Employees who worked for 5 or more years: ", worked_5years_df['Annual Salary'].mean())


# In[14]:


f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
sns.boxplot(curr_emp['Annual Salary'], ax=ax1)
sns.boxplot(quit_less_one_year_df['Annual Salary'], ax=ax2)
sns.boxplot(quit_less_two_year_df['Annual Salary'], ax=ax3)
sns.boxplot(quit_less_three_year_df['Annual Salary'], ax=ax4)
plt.tight_layout()


# In[15]:


f, (ax1, ax2) = plt.subplots(2)
sns.boxplot(worked_8years_df['Annual Salary'], ax=ax1)
sns.boxplot(worked_5years_df['Annual Salary'], ax=ax2)


# In[16]:


# 2 sample T-test on Annual salary between current employees and left employees

import scipy.stats as stats

stats.ttest_ind(curr_emp['Annual Salary'], quit_emp['Annual Salary'])


# In[17]:


# 2 sample T-test on Annual salary between employees who left within 3 years and employees who worked for more than 8 years

stats.ttest_ind(quit_less_three_year_df['Annual Salary'], worked_8years_df['Annual Salary'])


# In[18]:


# 2 sample T-test on Annual salary between current employees  and employees who worked for more than 8 years

stats.ttest_ind(curr_emp['Annual Salary'], worked_8years_df['Annual Salary'])


# - Average annual salary of current employees are higher than that of employees who quit within a year and two years by 14375 dollars and 13638 dollars respectively. However, the difference between that of current employees and that of employees who left after 3 years is slightly lower (13139 dollars).
# - Another interesting point is that the average annual salary of workers who worked for 5 years or longer / 8 years or longer is actually lower than that of current employees. This may demonstrate that higher salary may not be a major factor in people's decisions to stay in Family Services for a long time. But their annual mean salaries are still higher than those who left within 3 years (4627 dollars). 
# - Conducted 3 two pair T-test on Current Employees v.s. Terminated Employees, Current Employees v.s. Employees who worked 8+ years, and employees who left within 3 years v.s. employees who worked 8+ years but all of those tests showed statistically insignificant results. This is statistical proof that annual salary is not likely to be the statistically influential variable that causes turnover

# #### Supervisor ID

# In[19]:


quit_less_one_year_df['Supervisor ID'].value_counts()


# In[20]:


quit_less_two_year_df['Supervisor ID'].value_counts()


# In[21]:


quit_less_three_year_df['Supervisor ID'].value_counts()


# In[22]:


worked_8years_df['Supervisor ID'].value_counts()


# In[23]:


worked_5years_df['Supervisor ID'].value_counts()


# - One interesting point is that some names that appear for the list of supervisors who have the most employees leaving the workplace within 1,2,3 years also appear at the list of supervisors who have the most employees working for long years (e.g. 47W000739, MFR000598)
# - Those supervisors seem to lose a lot of supervisees within a short period of time but also have loyal workers who worked for a long time.

# #### EEOC Job Classification

# In[24]:


curr_emp['EEOC Job Classification'].value_counts()


# In[25]:


curr_emp['EEOC Job Classification'].value_counts()/sum(curr_emp['EEOC Job Classification'].value_counts())


# In[26]:


sns.countplot(y='EEOC Job Classification',data=curr_emp)
# plt.xticks(rotation='vertical') 


# In[27]:


print(quit_less_one_year_df['EEOC Job Classification'].value_counts())
print("\n")
print(quit_less_two_year_df['EEOC Job Classification'].value_counts())
print("\n")
print(quit_less_three_year_df['EEOC Job Classification'].value_counts())


# In[28]:


print(quit_less_one_year_df['EEOC Job Classification'].value_counts()/sum(quit_less_one_year_df['EEOC Job Classification'].value_counts()))
print("\n")
print(quit_less_two_year_df['EEOC Job Classification'].value_counts()/sum(quit_less_two_year_df['EEOC Job Classification'].value_counts()))
print("\n")
print(quit_less_three_year_df['EEOC Job Classification'].value_counts()/sum(quit_less_three_year_df['EEOC Job Classification'].value_counts()))


# In[29]:


sns.countplot(y='EEOC Job Classification',data=quit_less_one_year_df)


# In[30]:


sns.countplot(y='EEOC Job Classification',data=quit_less_two_year_df)


# In[31]:


sns.countplot(y='EEOC Job Classification',data=quit_less_three_year_df)


# In[32]:


worked_5years_df['EEOC Job Classification'].value_counts()


# In[33]:


worked_5years_df['EEOC Job Classification'].value_counts()/sum(worked_5years_df['EEOC Job Classification'].value_counts())


# In[34]:


worked_8years_df['EEOC Job Classification'].value_counts()


# In[35]:


worked_8years_df['EEOC Job Classification'].value_counts()/sum(worked_8years_df['EEOC Job Classification'].value_counts())


# In[36]:


f, (ax1, ax2) = plt.subplots(2)
sns.countplot(y='EEOC Job Classification', ax=ax1, data=worked_5years_df)
sns.countplot(y='EEOC Job Classification', ax=ax2, data=worked_8years_df)
plt.tight_layout()


# - For all six groups, Professional, Adminstrative Support Worker, and Service Worker job categories make up the biggest proportions.
# - When we compare the three groups of employees who left prematurely (after 1, 2, 3 years), it is notable that the proportion of professionals and Adminstrative Support Workers is higher in the 3rd year of leave than the 1st year of leave while the proportion of service workers is the opposite

# #### Job Title Description

# In[37]:


curr_emp['Job Title Description'].value_counts()


# In[38]:


quit_less_one_year_df['Job Title Description'].value_counts()


# In[39]:


quit_less_two_year_df['Job Title Description'].value_counts()


# In[40]:


quit_less_three_year_df['Job Title Description'].value_counts()


# In[41]:


worked_5years_df['Job Title Description'].value_counts()


# In[42]:


worked_8years_df['Job Title Description'].value_counts()


# - Among current employees, Group Leaders, IT Specialists, Group Aides, LMSW and Office Specialists make up the biggest proportions and they happen to be the top 4 occupations that worked the longest at Family Services
# - Group Aide, Group Leader, Program Assistant, LMSW, Office Specialists and Program Coordinators were the top 6 occupations that left Family Services within 3 years

# #### Scheduled Hours

# In[43]:


curr_emp['Scheduled Hours'].value_counts()


# In[44]:


print(quit_less_one_year_df['Scheduled Hours'].value_counts())
print("\n")
print(quit_less_two_year_df['Scheduled Hours'].value_counts())
print("\n")
print(quit_less_three_year_df['Scheduled Hours'].value_counts())


# In[45]:


print(quit_less_one_year_df['Scheduled Hours'].value_counts()/sum(quit_less_one_year_df['Scheduled Hours'].value_counts()))
print("\n")
print(quit_less_two_year_df['Scheduled Hours'].value_counts()/sum(quit_less_two_year_df['Scheduled Hours'].value_counts()))
print("\n")
print(quit_less_three_year_df['Scheduled Hours'].value_counts()/sum(quit_less_three_year_df['Scheduled Hours'].value_counts()))


# In[46]:


print(worked_5years_df['Scheduled Hours'].value_counts())
print("\n")
print(worked_8years_df['Scheduled Hours'].value_counts())


# In[47]:


print(worked_5years_df['Scheduled Hours'].value_counts()/sum(worked_5years_df['Scheduled Hours'].value_counts()))
print("\n")
print(worked_8years_df['Scheduled Hours'].value_counts()/sum(worked_8years_df['Scheduled Hours'].value_counts()))


# - Among employees who left within 1,2,3 years, 35 hours and 42.5 hours make up the biggest proportions(90%+ combined)
# - Most of the employees who worked for long years at Family Services are/were part-time employees working 10 hours, 20 hours, 11 hours, 14 hours and 24 hours per week (70%+ combined). Of course, 43% of the employees who worked for 8+ years worked 35 hours per week(full time workers)

# #### Ethnicity Proportion

# In[48]:


curr_emp['Ethnicity'].value_counts() / sum(curr_emp['Ethnicity'].value_counts().values)


# In[49]:


quit_less_one_year_df['Ethnicity'].value_counts() / sum(quit_less_one_year_df['Ethnicity'].value_counts().values)


# In[50]:


quit_less_two_year_df['Ethnicity'].value_counts() / sum(quit_less_two_year_df['Ethnicity'].value_counts().values)


# In[51]:


quit_less_three_year_df['Ethnicity'].value_counts() / sum(quit_less_three_year_df['Ethnicity'].value_counts().values)


# In[52]:


worked_5years_df['Ethnicity'].value_counts() / sum(worked_5years_df['Ethnicity'].value_counts().values)


# In[53]:


worked_8years_df['Ethnicity'].value_counts() / sum(worked_8years_df['Ethnicity'].value_counts().values)


# We can see that majority of employees at Family Services are non-Hispanic or Latino(80%) for all six groups. It is notable that the proportion of Hispanic or Lation employees is higher for the group of employees who quit within 3 years than that of current employees.

# #### Race Proportions

# In[54]:


curr_emp['Race'].value_counts() / sum(curr_emp['Race'].value_counts())


# In[55]:


quit_less_one_year_df['Race'].value_counts() / sum(quit_less_one_year_df['Race'].value_counts())


# In[56]:


quit_less_two_year_df['Race'].value_counts() / sum(quit_less_two_year_df['Race'].value_counts())


# In[57]:


quit_less_three_year_df['Race'].value_counts() / sum(quit_less_three_year_df['Race'].value_counts())


# In[58]:


worked_8years_df['Race'].value_counts() / sum(worked_8years_df['Race'].value_counts())


# - It is interesting to see how the proprotion of who quit decreases as times goes by (1st year to 2nd year to 3rd year) while the proportion of Hispanic employees who quit steadily increases(18% to 19% to 23%)
# - For employees who worked 8+ years at Family Services, 87% of them are Hispanic or Latino Employees which is a stark contrast to the race proportiosn we got for those who left Family Services prematurely

# #### Gender Proportion

# In[59]:


curr_emp['Gender'].value_counts() / sum(curr_emp['Gender'].value_counts().values)


# In[60]:


quit_less_one_year_df['Gender'].value_counts() / sum(quit_less_one_year_df['Gender'].value_counts().values)


# In[61]:


quit_less_two_year_df['Gender'].value_counts() / sum(quit_less_two_year_df['Gender'].value_counts().values)


# In[62]:


quit_less_three_year_df['Gender'].value_counts() / sum(quit_less_three_year_df['Gender'].value_counts().values)


# In[63]:


worked_5years_df['Gender'].value_counts() / sum(worked_5years_df['Gender'].value_counts().values)


# In[64]:


worked_8years_df['Gender'].value_counts() / sum(worked_8years_df['Gender'].value_counts().values)


# - Majority (75%+) of workers in Family Services identify themselves as females
# - Proportion of female employees who quit within “two” years and “three years” are about 5% and 3% lower than that of current employees. (76% v.s. 61% and 76% vs. 73%) respectively.
# - Gender Proportions seem to be mostly steady across different group regardless of whether you leave Family Services early or not

# #### Promotion

# In[65]:


curr_emp['Promotion'].value_counts() / sum(curr_emp['Promotion'].value_counts().values)


# In[66]:


quit_less_one_year_df['Promotion'].value_counts() / sum(quit_less_one_year_df['Promotion'].value_counts().values)


# In[67]:


quit_less_two_year_df['Promotion'].value_counts() / sum(quit_less_two_year_df['Promotion'].value_counts().values)


# In[68]:


quit_less_three_year_df['Promotion'].value_counts() / sum(quit_less_three_year_df['Promotion'].value_counts().values)


# In[69]:


worked_5years_df['Promotion'].value_counts() / sum(worked_5years_df['Promotion'].value_counts().values)


# In[70]:


worked_8years_df['Promotion'].value_counts() / sum(worked_8years_df['Promotion'].value_counts().values)


# This is previous analysis
# - 85%+ of employees who quit within the first few years don’t get promotions. This is natural because promotions are not given that often in the first few years.
# - 75% of the employees who worked 8+ years got promotions and this is higher than the proportion of current employees who got promotions. Promotions do seem to play a role in providing an incentive for employees to stay longer to a certain extent.

# #### Education Level

# In[71]:


curr_emp['Education Level Code'].value_counts() / sum(curr_emp['Education Level Code'].value_counts().values)


# In[134]:


pd.Series(curr_emp['Education Level Code'].value_counts() / sum(curr_emp['Education Level Code'].value_counts().values)).plot('barh')


# In[72]:


quit_less_one_year_df['Education Level Code'].value_counts() / sum(quit_less_one_year_df['Education Level Code'].value_counts().values)


# In[73]:


quit_less_two_year_df['Education Level Code'].value_counts() / sum(quit_less_two_year_df['Education Level Code'].value_counts().values)


# In[74]:


quit_less_three_year_df['Education Level Code'].value_counts() / sum(quit_less_three_year_df['Education Level Code'].value_counts().values)


# In[135]:


pd.Series(quit_less_three_year_df['Education Level Code'].value_counts() /sum(quit_less_three_year_df['Education Level Code'].value_counts().values)).plot('barh')


# In[75]:


worked_5years_df['Education Level Code'].value_counts() / sum(worked_5years_df['Education Level Code'].value_counts().values)


# In[76]:


worked_8years_df['Education Level Code'].value_counts() / sum(worked_8years_df['Education Level Code'].value_counts().values)


# In[136]:


pd.Series(worked_8years_df['Education Level Code'].value_counts() / sum(worked_8years_df['Education Level Code'].value_counts().values)).plot('barh')


# #### FLSA Code

# In[77]:


curr_emp['FLSA Code'].value_counts()/sum(curr_emp['FLSA Code'].value_counts())


# In[78]:


quit_less_one_year_df['FLSA Code'].value_counts()/sum(quit_less_one_year_df['FLSA Code'].value_counts())


# In[79]:


quit_less_two_year_df['FLSA Code'].value_counts()/sum(quit_less_two_year_df['FLSA Code'].value_counts())


# In[80]:


quit_less_three_year_df['FLSA Code'].value_counts()/sum(quit_less_three_year_df['FLSA Code'].value_counts())


# In[81]:


worked_5years_df['FLSA Code'].value_counts()/sum(worked_5years_df['FLSA Code'].value_counts())


# In[82]:


worked_8years_df['FLSA Code'].value_counts()/sum(worked_8years_df['FLSA Code'].value_counts())


# - Proportion of Non-Exempt employees(70%+) are overwhelmingly higher for employees who left Family Services in the first few years. This suggests most of the employees who quit in the first few years are not in professional, management, executive, supervisory positions. But the proportion non-exempt employees is lower than that of current employees (70% ish < 81%).
# - But 94% of employees who worked 8+ years are/were non-exempt employees and this proportion of non-exempt employees is 13% higher than that of current employees.

# #### Benefits Eligibility Class Description

# In[83]:


curr_emp['Benefits Eligibility Class Description'].value_counts()


# In[84]:


quit_less_one_year_df['Benefits Eligibility Class Description'].value_counts()


# In[140]:


quit_less_one_year_df['Benefits Eligibility Class Description'].value_counts()/sum(quit_less_one_year_df['Benefits Eligibility Class Description'].value_counts())


# In[85]:


quit_less_two_year_df['Benefits Eligibility Class Description'].value_counts()


# In[139]:


quit_less_two_year_df['Benefits Eligibility Class Description'].value_counts()/sum(quit_less_two_year_df['Benefits Eligibility Class Description'].value_counts())


# In[86]:


quit_less_three_year_df['Benefits Eligibility Class Description'].value_counts()


# In[138]:


quit_less_three_year_df['Benefits Eligibility Class Description'].value_counts()/ sum(quit_less_three_year_df['Benefits Eligibility Class Description'].value_counts())


# In[87]:


worked_5years_df['Benefits Eligibility Class Description'].value_counts()


# In[88]:


worked_8years_df['Benefits Eligibility Class Description'].value_counts()


# In[137]:


worked_8years_df['Benefits Eligibility Class Description'].value_counts() / sum(worked_8years_df['Benefits Eligibility Class Description'].value_counts())


# - Solid Majority of employees who work long years (8+) are full-time regular workers. 
# - Higher proportions of part-time regular employees leave over the years (4.1% => 4.5% => 4.9%) although the increment isn’t that big

# #### Home Department

# In[89]:


curr_emp['Home Department Description'].value_counts()


# In[90]:


quit_less_three_year_df['Home Department Description'].value_counts()


# In[91]:


worked_8years_df['Home Department Description'].value_counts()


# - Elvasp Pt department has a lot of employees for all three groups
# - Pmhc Ft, Youth Activities, BMHC and Battered Womens Svcs Department seem to have a lot of employees who leave in the first 3 years and relatively have less employees who work 8+ years. These departments should receive more attention to prevent turnovers.

# #### Age

# In[141]:


sns.distplot(curr_emp['Age'])


# In[144]:


max(curr_emp[curr_emp['Age']!=0.0]['Age']), min(curr_emp[curr_emp['Age']!=0.0]['Age'])


# In[145]:


np.mean(curr_emp[curr_emp['Age']!=0.0]['Age']), np.median(curr_emp[curr_emp['Age']!=0.0]['Age'])


# In[146]:


np.std(curr_emp[curr_emp['Age']!=0.0]['Age'])


# In[97]:


sns.distplot(quit_less_three_year_df['Age'])


# In[98]:


sns.distplot(quit_less_two_year_df['Age'])


# In[99]:


sns.distplot(quit_less_one_year_df['Age'])


# In[100]:


max(quit_less_three_year_df['Age']),min(quit_less_three_year_df['Age'])


# In[101]:


np.mean(quit_less_three_year_df['Age']), np.median(quit_less_three_year_df['Age'])


# In[102]:


np.std(quit_less_three_year_df['Age'])


# In[103]:


max(quit_less_two_year_df['Age']),min(quit_less_two_year_df['Age'])


# In[104]:


np.mean(quit_less_two_year_df['Age']), np.median(quit_less_two_year_df['Age'])


# In[105]:


np.std(quit_less_two_year_df['Age'])


# In[106]:


max(quit_less_one_year_df['Age']),min(quit_less_one_year_df['Age'])


# In[107]:


np.mean(quit_less_one_year_df['Age']), np.median(quit_less_one_year_df['Age'])


# In[108]:


np.std(quit_less_one_year_df['Age'])


# In[109]:


sns.distplot(worked_8years_df[worked_8years_df['Age']!=0.0]['Age'])


# In[110]:


max(worked_8years_df[worked_8years_df['Age']!=0.0]['Age']), min(worked_8years_df[worked_8years_df['Age']!=0.0]['Age']), 


# In[111]:


np.mean(worked_8years_df[worked_8years_df['Age']!=0.0]['Age']), np.median(worked_8years_df[worked_8years_df['Age']!=0.0]['Age'])


# In[112]:


np.std(worked_8years_df[worked_8years_df['Age']!=0.0]['Age'])


# In[113]:


max(worked_5years_df[worked_5years_df['Age']!=0.0]['Age']), min(worked_5years_df[worked_5years_df['Age']!=0.0]['Age']), 


# In[114]:


np.mean(worked_5years_df[worked_5years_df['Age']!=0.0]['Age']), np.median(worked_5years_df[worked_5years_df['Age']!=0.0]['Age'])


# In[115]:


np.std(worked_5years_df[worked_5years_df['Age']!=0.0]['Age'])


# - Employees who worked long years tended to have similar average age to that of current employees (Late 40s)
# - Employees who quit in their first few years were, on average, on their early/mid 40s.
# - Thus, employees who quit early were, in general, slightly younger than the average workforce of Family Services or those who worked for a long time (8+ years)

# #### Regular Pay Rate Amount

# In[147]:


curr_emp['Regular Pay Rate Amount'].value_counts()/sum(curr_emp['Regular Pay Rate Amount'].value_counts())


# In[117]:


sns.distplot(curr_emp['Regular Pay Rate Amount'])


# In[148]:


quit_less_three_year_df['Regular Pay Rate Amount'].value_counts()/sum(quit_less_three_year_df['Regular Pay Rate Amount'].value_counts())


# In[119]:


sns.distplot(quit_less_three_year_df['Regular Pay Rate Amount'])


# In[151]:


worked_8years_df['Regular Pay Rate Amount'].value_counts()/sum(worked_8years_df['Regular Pay Rate Amount'].value_counts())


# In[150]:


sns.distplot(worked_8years_df['Regular Pay Rate Amount'])


# In[153]:


for data in [curr_emp, quit_less_one_year_df, quit_less_two_year_df, quit_less_three_year_df, worked_5years_df, worked_8years_df]:
    print(max(data['Regular Pay Rate Amount']), min(data['Regular Pay Rate Amount']), 
          np.mean(data['Regular Pay Rate Amount']), np.median(data['Regular Pay Rate Amount']))


# In[155]:


stats.ttest_ind(curr_emp['Regular Pay Rate Amount'], quit_emp['Regular Pay Rate Amount'])


# In[156]:


stats.ttest_ind(worked_8years_df['Regular Pay Rate Amount'], quit_less_three_year_df['Regular Pay Rate Amount'])


# In[157]:


stats.ttest_ind(curr_emp['Regular Pay Rate Amount'], quit_less_three_year_df['Regular Pay Rate Amount'])


# - Distribution of regular pay rate amount for current employees is pretty skewed to the right with a long tail towards the right side.
# - Mean Pay Rate and Median Pay Rate for employees who worked 8+ years are both lower than those of employees who left in the first few years
# - T-test on mean regular pay amount difference between current employees and all employees who left shows there is a statistically significant evidence that current employees, on average, receive higher regular pay rate amounts than employees who left Family Services.

# #### Hire Date

# In[158]:


sns.countplot(y="Hire Month", hue='Left', data=data)A


# In[161]:


f, ax = plt.subplots(figsize=(10, 8))
sns.countplot(y="Hire Year", hue='Left', data=data)


# - A lot of employees left in April, May, September and October
# - A lot of current employees who are still working at Family Services were hired in April
# - From 2001 to 2008, a lot of employees left Family Services and that's also when a lot of hiring for the current employees took place as well

# #### Linear Correlation between numerical variables and left variable(whether employee has quit or not)

# In[172]:


## Linear Correlation between numerical variables and left variable(whether employee has quit or not)

corrmat = data.drop(labels=['Race Code','Unnamed: 0'], axis=1).corr()

corrmat


# In[173]:


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin = -1, vmax = 1, square=True, annot=True,fmt='.2f')

corr_list = pd.DataFrame(corrmat['Left'].sort_values().drop('Left'))
corr_list


# - No variable seems to have a very strong linear correlation with the Left variable. Some variables that still have some weak correlation with the left variable are annual salary and years of service
# - Scheduled Hours and Years of service have very strong negative correlation
# - Annual Salary has pretty strong correlation with Age and Regular Pay Rate Amount
# - Hire Year has pretty strong negative correlation with Age and Regular Pay Amount and positive correlation with Scheduled Hours. I guess this makes sense because the earlier you have been hired, the more likely you are to be older and have higher regular pay amount

# In[175]:


sns.pairplot(data,vars=['Age','Scheduled Hours', 'Regular Pay Rate Amount', 'Annual Salary', 'Years of Service'],hue='Left')

