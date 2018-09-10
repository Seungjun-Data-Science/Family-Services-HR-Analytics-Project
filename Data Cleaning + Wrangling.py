{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Cleaning + Wrangling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing Libraries, Merging data sets, Handling duplicates and weird labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data = pd.read_excel('hr_data_final.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "data2 = pd.read_excel('curr_employee.xls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "data3 = pd.read_excel('turnover.xls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data4 = pd.read_excel('degree_job_class_filled.xls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping Duplicate Columns from data\n",
    "data = data.loc[:,~data.columns.duplicated()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "data = data.drop_duplicates(subset=['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "data2 = data2.drop_duplicates(subset=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "data3 = data3.drop_duplicates(subset=['Gender','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data = pd.merge(data3, data, on=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'],how='outer')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true,
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+\"_y\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():\n",
    "    del fin_data[col]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data = fin_data.rename(columns={col:col[:-2]})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Deleting unnecessary columns in data\n",
    "for col in ['Education Level Description.1','Annual Salary.1','Pay Frequency.1','Regular Pay Rate Amount.1','Education Level Description']:\n",
    "    del fin_data[col]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "fin_data = fin_data.drop_duplicates(subset=['Ethnicity','Gender','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fin_data = pd.merge(fin_data, data2,on=['Age','Annual Salary','Ethnicity','Gender','Position Effective Date'], how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+\"_y\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():\n",
    "    del fin_data[col]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data = fin_data.rename(columns={col:col[:-2]})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Unnamed: 6']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Business Unit Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data4['Education Level Code'] = data4['Education Level Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del data4['Education Level Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data = pd.merge(fin_data,data4[['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date',\n",
    "                                    'Education Level Code','EEOC Job Classification']], \n",
    "                    on=['Tax ID','Ethnicity','Gender','Annual Salary','Age','Position Effective Date'], how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data.loc[fin_data[col].isnull(),col] = fin_data[col[:-2]+\"_y\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_y')].tolist():\n",
    "    del fin_data[col]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "for col in fin_data.columns[fin_data.columns.str.endswith('_x')].tolist():\n",
    "    fin_data = fin_data.rename(columns={col:col[:-2]})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 3304 entries, 0 to 3303\n",
      "Data columns (total 33 columns):\n",
      "Ethnicity                                 3211 non-null object\n",
      "Annual Salary                             3304 non-null float64\n",
      "Position Effective Date                   3301 non-null datetime64[ns]\n",
      "Position Start Date                       1773 non-null datetime64[ns]\n",
      "Hire Date                                 3302 non-null datetime64[ns]\n",
      "Termination Date                          2652 non-null datetime64[ns]\n",
      "Years of Service                          1946 non-null float64\n",
      "Years in Current Position                 1946 non-null float64\n",
      "Promotion Check                           1946 non-null object\n",
      "Promotion                                 1946 non-null object\n",
      "Race Code                                 2883 non-null float64\n",
      "EEOC Job Classification                   1800 non-null object\n",
      "Gender                                    3304 non-null object\n",
      "Supervisor ID                             1101 non-null object\n",
      "Current Date                              1946 non-null datetime64[ns]\n",
      "Age                                       3233 non-null object\n",
      "Job Change Reason Code                    1363 non-null object\n",
      "Race Description                          1191 non-null object\n",
      "Race                                      1191 non-null object\n",
      "FLSA Code                                 1363 non-null object\n",
      "Home Department Code                      622 non-null float64\n",
      "Home Department Description               622 non-null object\n",
      "Job Title Description                     647 non-null object\n",
      "Payroll Company Code                      1363 non-null object\n",
      "Reports To First Name                     305 non-null object\n",
      "Reports To Last Name                      305 non-null object\n",
      "Education Level Code                      422 non-null object\n",
      "Scheduled Hours                           432 non-null float64\n",
      "Tax ID                                    1363 non-null object\n",
      "Regular Pay Rate Amount                   1363 non-null float64\n",
      "Pay Frequency                             1363 non-null object\n",
      "Benefits Eligibility Class Code           433 non-null object\n",
      "Benefits Eligibility Class Description    433 non-null object\n",
      "dtypes: datetime64[ns](5), float64(7), object(21)\n",
      "memory usage: 877.6+ KB\n"
     ]
    }
   ],
   "source": [
    "fin_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "fin_data = fin_data.drop_duplicates(subset=['Gender','Ethnicity','Annual Salary','Age','Position Effective Date','Years of Service'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 3304 entries, 0 to 3303\n",
      "Data columns (total 33 columns):\n",
      "Ethnicity                                 3211 non-null object\n",
      "Annual Salary                             3304 non-null float64\n",
      "Position Effective Date                   3301 non-null datetime64[ns]\n",
      "Position Start Date                       1773 non-null datetime64[ns]\n",
      "Hire Date                                 3302 non-null datetime64[ns]\n",
      "Termination Date                          2652 non-null datetime64[ns]\n",
      "Years of Service                          1946 non-null float64\n",
      "Years in Current Position                 1946 non-null float64\n",
      "Promotion Check                           1946 non-null object\n",
      "Promotion                                 1946 non-null object\n",
      "Race Code                                 2883 non-null float64\n",
      "EEOC Job Classification                   1800 non-null object\n",
      "Gender                                    3304 non-null object\n",
      "Supervisor ID                             1101 non-null object\n",
      "Current Date                              1946 non-null datetime64[ns]\n",
      "Age                                       3233 non-null object\n",
      "Job Change Reason Code                    1363 non-null object\n",
      "Race Description                          1191 non-null object\n",
      "Race                                      1191 non-null object\n",
      "FLSA Code                                 1363 non-null object\n",
      "Home Department Code                      622 non-null float64\n",
      "Home Department Description               622 non-null object\n",
      "Job Title Description                     647 non-null object\n",
      "Payroll Company Code                      1363 non-null object\n",
      "Reports To First Name                     305 non-null object\n",
      "Reports To Last Name                      305 non-null object\n",
      "Education Level Code                      422 non-null object\n",
      "Scheduled Hours                           432 non-null float64\n",
      "Tax ID                                    1363 non-null object\n",
      "Regular Pay Rate Amount                   1363 non-null float64\n",
      "Pay Frequency                             1363 non-null object\n",
      "Benefits Eligibility Class Code           433 non-null object\n",
      "Benefits Eligibility Class Description    433 non-null object\n",
      "dtypes: datetime64[ns](5), float64(7), object(21)\n",
      "memory usage: 877.6+ KB\n"
     ]
    }
   ],
   "source": [
    "fin_data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Missing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data['Age'] = fin_data['Age'].fillna(np.median(fin_data['Age']))\n",
    "fin_data['Ethnicity'] = fin_data['Ethnicity'].fillna('Hispanic or Latino')\n",
    "fin_data['Race'] = fin_data['Race'].fillna('Hispanic or Latino')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have same job classification, assumed they work same number of hours\n",
    "fin_data['Scheduled Hours'] = fin_data.groupby('EEOC Job Classification')['Scheduled Hours'].bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.0"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fin_data['Scheduled Hours'].value_counts().index[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# new race code 10.0 denoting \"Hispanic or Lationo\"\n",
    "fin_data['Race Code'] = fin_data['Race Code'].fillna(10.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# If two employees have same home department, Job Title Description and number of work hours, assumed their regular pay amount is equal\n",
    "fin_data['Regular Pay Rate Amount'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Scheduled Hours','Home Department Description'])['Regular Pay Rate Amount'].bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Fill in the still missing values with the most frequent value of that variable\n",
    "fin_data['Scheduled Hours'] = fin_data['Scheduled Hours'].fillna(fin_data['Scheduled Hours'].value_counts().index[0])\n",
    "fin_data['Regular Pay Rate Amount'] = fin_data['Regular Pay Rate Amount'].\\\n",
    "fillna(fin_data['Regular Pay Rate Amount'].value_counts().index[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "del fin_data['Position Start Date']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Promotion Check']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "del fin_data['Home Department Code']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If position effective data doesn't match hiring date, means the employee has been promoted\n",
    "fin_data['check'] = np.where(fin_data['Position Effective Date']!=fin_data['Hire Date'], 'Yes', 'No')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# If position effective data doesn't match hiring date, means the employee has been promoted\n",
    "fin_data.loc[fin_data['Promotion'].isnull(),'Promotion'] =\\\n",
    "fin_data['check']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['check']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Race Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have same home department, Job Classification and number of work hours, assumed their Supervisor is the same\n",
    "fin_data['Supervisor ID'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Home Department Description','Scheduled Hours'])['Supervisor ID'].bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fin_data['Supervisor Name'] = fin_data['Reports To First Name'] + \" \" + fin_data['Reports To Last Name']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the smae supervisor ID, then the supervisor's name would be the same\n",
    "fin_data['Supervisor Name'] =\\\n",
    "fin_data.groupby(['Supervisor ID'])['Supervisor Name'].bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have same anuual salary, scheduled number of hours and regular pay amount, \n",
    "# assumed their job classification would be the same\n",
    "fin_data['EEOC Job Classification'] =\\\n",
    "fin_data.groupby(['Annual Salary','Scheduled Hours','Regular Pay Rate Amount'])['EEOC Job Classification'].bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Current Date']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Job Change Reason Code']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Reports To First Name']\n",
    "del fin_data['Reports To Last Name']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Benefits Eligibility Class Code']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same Job Classification, number of work hours, regular pay amount, and annual salary,\n",
    "# assumed their job titles would be the same\n",
    "fin_data['Job Title Description'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Scheduled Hours','Regular Pay Rate Amount','Annual Salary'])['Job Title Description'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Pay Frequency']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same Job Classification, job description, regular pay amount, and annual salary, \n",
    "# assumed FLSA Code would be the same\n",
    "fin_data['FLSA Code'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Job Title Description','Regular Pay Rate Amount','Annual Salary'])['FLSA Code'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "fin_data['FLSA Code'] = fin_data['FLSA Code'].replace({' ':np.nan})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same Job Classification, job description, regular pay amount, and annual salary, \n",
    "# assumed FLSA Code would be the same\n",
    "fin_data['FLSA Code'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Job Title Description','Regular Pay Rate Amount','Annual Salary'])['FLSA Code'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same Annual Salary, job description, and regular pay amount, assumed Education Level would be the similar(same)\n",
    "fin_data['Education Level Code'] =\\\n",
    "fin_data.groupby(['Job Title Description','Regular Pay Rate Amount','Annual Salary'])['Education Level Code'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same supervisor, job classification and job description, assumed Home Department would be the same\n",
    "fin_data['Home Department Description'] =\\\n",
    "fin_data.groupby(['Supervisor ID','EEOC Job Classification','Job Title Description'])['Home Department Description'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Column that indicates whether this employee is still in Family Services or has been terminated(left)\n",
    "fin_data['Left'] = np.where(fin_data['Termination Date'].notnull(), 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill null with today's timestamp\n",
    "from pandas import Timestamp\n",
    "fin_data['Termination Date'] = fin_data['Termination Date'].fillna(Timestamp.today())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Years of service in timedelta\n",
    "fin_data['timedelta'] = fin_data['Termination Date'] - fin_data['Hire Date']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Converting Years of service in timedelta into number of years\n",
    "fin_data['timedelta'] = (fin_data['timedelta'].dt.days)/365"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data.loc[fin_data['Years of Service'].isnull(),'Years of Service']=\\\n",
    "fin_data['timedelta']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['timedelta']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Termination Date']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# If two employees have the same job classification, job description, and annual salary assumed Benefits Eligibility would be the same\n",
    "fin_data['Benefits Eligibility Class Description'] =\\\n",
    "fin_data.groupby(['EEOC Job Classification','Job Title Description','Annual Salary'])['Benefits Eligibility Class Description'].\\\n",
    "bfill().ffill()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['Years in Current Position']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Filling in missing information of categorical variables with \"Not Reported\"\n",
    "for col in ['Supervisor ID','Supervisor Name','FLSA Code','Home Department Description','Job Title Description',\n",
    "           'Payroll Company Code','Education Level Code','Education Level Code',\n",
    "            'Benefits Eligibility Class Description']:\n",
    "    fin_data[col] = fin_data[col].fillna('Not Reported')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data['EEOC Job Classification'] = fin_data['EEOC Job Classification'].replace({'Not reported': 'Not Reported'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data['FLSA Code'] = fin_data['FLSA Code'].replace({' ': 'Not Reported'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data = fin_data.reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Dropping duplicates and keeping the first ones\n",
    "data = data.drop_duplicates(subset=['Age','Race','Ethnicity','Annual Salary','Scheduled Hours','Regular Pay Rate Amount'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "del fin_data['index']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Changing datatype of Hire Date into \"datetime\" for convenience\n",
    "fin_data['Hire Date'] = pd.to_datetime(fin_data['Hire Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fin_data['Hire Year'] = data['Hire Date'].dt.year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fin_data['Hire Year'] = fin_data['Hire Year'].astype('int64',errors='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "fin_data['Hire Month'] = fin_data['Hire Date'].dt.month"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "fin_data['Hire Month'] = fin_data['Hire Month'].astype('int64',errors='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 3304 entries, 0 to 3303\n",
      "Data columns (total 25 columns):\n",
      "Ethnicity                                 3304 non-null object\n",
      "Annual Salary                             3304 non-null float64\n",
      "Position Effective Date                   3301 non-null datetime64[ns]\n",
      "Hire Date                                 3302 non-null datetime64[ns]\n",
      "Years of Service                          3303 non-null float64\n",
      "Promotion                                 3304 non-null object\n",
      "Race Code                                 3304 non-null float64\n",
      "EEOC Job Classification                   3304 non-null object\n",
      "Gender                                    3304 non-null object\n",
      "Supervisor ID                             3304 non-null object\n",
      "Age                                       3304 non-null float64\n",
      "Race                                      3304 non-null object\n",
      "FLSA Code                                 3304 non-null object\n",
      "Home Department Description               3304 non-null object\n",
      "Job Title Description                     3304 non-null object\n",
      "Payroll Company Code                      3304 non-null object\n",
      "Education Level Code                      3304 non-null object\n",
      "Scheduled Hours                           3304 non-null float64\n",
      "Tax ID                                    1363 non-null object\n",
      "Regular Pay Rate Amount                   3304 non-null float64\n",
      "Benefits Eligibility Class Description    3304 non-null object\n",
      "Supervisor Name                           3304 non-null object\n",
      "Left                                      3304 non-null int32\n",
      "Hire Year                                 1217 non-null float64\n",
      "Hire Month                                3302 non-null float64\n",
      "dtypes: datetime64[ns](2), float64(8), int32(1), object(14)\n",
      "memory usage: 632.5+ KB\n"
     ]
    }
   ],
   "source": [
    "fin_data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Saving into CSV Files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Save cleaned data as a new file\n",
    "\n",
    "fin_data.to_csv('clean_data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Current Employees\n",
    "curr_emp = fin_data[fin_data['Left']==0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "curr_emp.to_csv('curr_emp.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Terminated Employees\n",
    "left_emp = fin_data[fin_data['Left']==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "left_emp.to_csv('left_emp.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3304, 25) (652, 25) (2652, 25)\n"
     ]
    }
   ],
   "source": [
    "print(fin_data.shape, curr_emp.shape, left_emp.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "652 current employees and 2652 terminated employees"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
