import pandas as pd
#import data and generate the description
pd.set_option('display.max_columns',999)
df_07_11 = pd.read_csv('https://resources.lendingclub.com/LoanStats3a.csv.zip', compression='zip', skiprows=[0])
df_12_13 = pd.read_csv('https://resources.lendingclub.com/LoanStats3b.csv.zip', compression='zip', skiprows=[0])
df_14 = pd.read_csv('https://resources.lendingclub.com/LoanStats3c.csv.zip', compression='zip', skiprows=[0])
df_15 = pd.read_csv('https://resources.lendingclub.com/LoanStats3d.csv.zip', compression='zip', skiprows=[0])
df_16Q1 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2016Q1.csv.zip', compression='zip', skiprows=[0])
df_16Q2 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2016Q2.csv.zip', compression='zip', skiprows=[0])
df_16Q3 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2016Q3.csv.zip', compression='zip', skiprows=[0])
df_16Q4 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2016Q4.csv.zip', compression='zip', skiprows=[0])
df_17Q1 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2017Q1.csv.zip', compression='zip', skiprows=[0])
df_17Q2 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2017Q2.csv.zip', compression='zip', skiprows=[0])
df_17Q3 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2017Q3.csv.zip', compression='zip', skiprows=[0])
df_17Q4 = pd.read_csv('https://resources.lendingclub.com/LoanStats_2017Q4.csv.zip', compression='zip', skiprows=[0])

df_all=pd.concat([df_07_11,df_12_13,df_14,df_15,df_16Q1,df_16Q2,df_16Q3,df_16Q4,df_17Q1,df_17Q2,df_17Q3,df_17Q4])

df_all.describe()
df_all['default'] = 0
df_all['default'][(df_all['chargeoff_within_12_mths'] > 0)|((df_all['num_tl_120dpd_2m']>0)&(df_all['delinq_amnt']>50))] = 1


import matplotlib.pyplot as plt
import seaborn as sns
df_all['issue_d'].describe(include='all')
df_all['issue_y']=df_all['issue_d'].str[-2:]
df_all['issue_y'].describe(include='all')
#count default trend
df_all=df_all.sort_values(by='issue_y')
sns.barplot(x='issue_y',y='default',data=df_all)
#count customer number
sns.countplot(x='issue_y',data=df_all)
sns.barplot(x='issue_y',y='avg_cur_bal',data=df_all)
sns.barplot(x='issue_y',y='tot_cur_bal',data=df_all)
#boxplot
fig, ax = plt.subplots(2, 2)
sns.boxplot(x="default", y="avg_cur_bal", data=df_all)
sns.boxplot(x="default", y="tot_cur_bal", data=df_all)
sns.boxplot(x="default", y="loan_amnt", data=df_all)

importance=pd.read_csv('/Users/renfeigao/Desktop/lending club/importance.csv')
importance.info()
df_all.describe()

for x in importance['Unnamed: 0'][importance['Importance']>=0.01]:
   try:
       df_all[x].plot.hist()
   except:
        print(x)

df_all['revol_util'].plot.hist()

df_all['revol_util']
