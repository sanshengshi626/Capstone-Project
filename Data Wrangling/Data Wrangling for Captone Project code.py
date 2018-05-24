import pandas as pd

###################################loading data###########################################
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

pd.set_option('display.max_columns',999)
pd.set_option('display.max_rows',999)

##################################data information########################################
print(df_all.head())
df_all.describe(include='all')
df_all.info(verbose=True, null_counts=True)
df_all.describe(include='all').to_csv('/Users/renfeigao/Documents/GitHub/Capstone-Project/Data Wrangling/describe.csv')
df_all.dtypes.to_csv('/Users/renfeigao/Documents/GitHub/Capstone-Project/Data Wrangling/type.csv')
#################################define default logic####################################
df_all['default'] = 0
df_all['default'][(df_all['chargeoff_within_12_mths'] > 0)|((df_all['num_tl_120dpd_2m']>0)&(df_all['delinq_amnt']>50))] = 1
df_all.describe(include='all')
#check if the logic is good to define
print(df_all[['default','chargeoff_within_12_mths','num_tl_120dpd_2m','delinq_amnt']]
[(df_all['chargeoff_within_12_mths'] > 0)|((df_all['num_tl_120dpd_2m']>0)
&(df_all['delinq_amnt']>50))])
##################################check by year and drop unreliable data????################
df_all['issue_y']=pd.to_numeric(df_all['issue_d'].str[-2:])   
df_all.groupby(['issue_y']).sum()
#many information are missing beofre 2012, so drop the data before 2012
df_new=df_all[df_all['issue_y']>=12]
df_new.info(verbose=True, null_counts=True)
df_new.groupby(['issue_y']).sum()
#drop variables with all missing
df_new=df_new.drop(columns=['id','member_id','url'])
#################################fill missing variables####################
#for co borrower information, no info before year of 2017, so fill all with 0 ?????check
df_bf_2017=df_new[df_new['issue_y']<17]
for col in df_new.columns:
    if col[:3]=='sec':
        df_bf_2017[col]=df_bf_2017[col].fillna(0)
df_bf_2017['revol_bal_joint']=df_bf_2017['revol_bal_joint'].fillna(0)
df_new=pd.concat([df_bf_2017,df_new[df_new['issue_y']>=17]])
#some information are not available before 2015, fill 0
df_bf_2015=df_new[df_new['issue_y']<15]
df_bf_2015['annual_inc_joint']=df_bf_2015['annual_inc_joint'].fillna(0)
df_bf_2015['dti_joint']=df_bf_2015['dti_joint'].fillna(0)
df_bf_2015['open_acc_6m']=df_bf_2015['open_acc_6m'].fillna(0)
df_bf_2015['open_act_il']=df_bf_2015['open_act_il'].fillna(0)
df_bf_2015['open_il_12m']=df_bf_2015['open_il_12m'].fillna(0)
df_bf_2015['open_il_24m']=df_bf_2015['open_il_24m'].fillna(0)
df_bf_2015['mths_since_rcnt_il']=df_bf_2015['mths_since_rcnt_il'].fillna(0)
df_bf_2015['total_bal_il']=df_bf_2015['total_bal_il'].fillna(0)
df_bf_2015['il_util']=df_bf_2015['il_util'].fillna(0)
df_bf_2015['open_rv_12m']=df_bf_2015['open_rv_12m'].fillna(0)
df_bf_2015['open_rv_24m']=df_bf_2015['open_rv_24m'].fillna(0)
df_bf_2015['max_bal_bc']=df_bf_2015['max_bal_bc'].fillna(0)
df_bf_2015['all_util']=df_bf_2015['all_util'].fillna(0)
df_bf_2015['inq_fi']=df_bf_2015['inq_fi'].fillna(0)
df_bf_2015['total_cu_tl']=df_bf_2015['total_cu_tl'].fillna(0)
df_bf_2015['inq_last_12m']=df_bf_2015['inq_last_12m'].fillna(0)
df_new=pd.concat([df_bf_2015,df_new[df_new['issue_y']>=15]])

#########################check outliers########################################
#convert date variables    
import datetime as dt
def convert_day(date):
    df_all[date]=pd.to_datetime(df_all[date])
    td=pd.to_datetime('today')
    df_all[date]=df_all[date].apply(lambda x: td-x)


convert_day('issue_d')
convert_day('last_pymnt_d')
convert_day('next_pymnt_d')
convert_day('last_credit_pull_d')
convert_day('hardship_start_date')
convert_day('hardship_end_date')
convert_day('payment_plan_start_date')
convert_day('debt_settlement_flag_date')
convert_day('settlement_date')
convert_day('earliest_cr_line')
df_all['issue_d'].head()
df_all.describe(include='all')

#drop emp_title
import numpy
numpy.set_printoptions(threshold=numpy.nan)
df_all['emp_title'].nunique()
df_all['emp_title'].unique()
df_all=df_all.drop(columns=['emp_title'])

#convert string numeric variables to numbers
df_all['int_rate']=pd.to_numeric(df_all['int_rate'].str.replace('%', ''))
df_all['revol_util']=pd.to_numeric(df_all['revol_util'].str.replace('%', ''))
df_all['zip_code']=pd.to_numeric(df_all['zip_code'].str.replace('xx', ''))
#delete unvalid accounts
df_all=df_all[df_all['loan_amnt'].notnull()]
df_all.info(verbose=True, null_counts=True)
#one hot encoding the rest of the category variables
df_all=pd.get_dummies(df_all)

df_all.describe(include='all')

#fill missing value
df_all=df_all.fillna(0)
df_all.head()

#split data and dependece variable
default=df_all['default']
df_all_train=df_all.drop(columns=['default','chargeoff_within_12_mths','num_tl_120dpd_2m','delinq_amnt'])
df_all_train.dtypes.to_csv('/Users/renfeigao/Desktop/lending club/type_after.csv')
#first round model using radam forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(df_all_train, default);
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), df_all_train.columns), 
             reverse=True))
importances=rf.feature_importances_
importances=pd.DataFrame(importances, index=df_all_train.columns,columns=["Importance"]).sort_values(by=["Importance"],ascending=False)
importances.info()
importances.to_csv('/Users/renfeigao/Desktop/lending club/importance.csv')

