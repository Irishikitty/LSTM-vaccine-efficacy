import pandas as pd
import numpy as np
import tensorflow as tf


# CLEAN ---------------------------------------------------------------
df = pd.read_csv('prepared_data_3outcome_weekly.csv')
df['Visits'].value_counts()
df['Comorb'].value_counts()
df['AGE90'].value_counts()
df['GENDER'].value_counts()
df['RACE'].value_counts()
visits = {'0-4':0,'5-9':1,'10-19':2,'20-50':3,'>50':4}
comorbidity = {'0':0,'1-2':1,'3-4':2,'>=5':3}
gender = {'Female':0,'Male':1}
race = {'White or Caucasian':0, 'Black or African American':1, 'Other':2}
df['Visits'] = df['Visits'].replace(visits)
df['Comorb'] = df['Comorb'].replace(comorbidity)
df['GENDER'] = df['GENDER'].replace(gender)
df['RACE'] = df['RACE'].replace(race)
df.rename(columns={"newid":"ID","padding":"censor_prev"}, inplace=True)

print('-'*30)
unique_id = df['ID'].unique()
print(f'Number of patients:   {len(unique_id)}')


### NOTE: if raw data does not include 'end date', then the following is not needed. #########
# if time==104, event==0, VaxDose reaches its second maximum//VacDose在T=104时可能+1
# second_max_vax_dose_per_patient = df.groupby("ID")["VaxDose"].apply(lambda x: x.nlargest(2).iloc[-1])
# second_max_vax_dose_per_patient.name = 'SecondMaxVacN'
# df = df.merge(second_max_vax_dose_per_patient,on='ID')
# filtered_df = df[~((df["Time"] == 104) & \
#     (df["event"] == 0) & \
#     (df["VaxDose"] == df['SecondMaxVacN']))]
# df = filtered_df.drop('SecondMaxVacN',axis=1)


TOTAL_TIME = df['Time'].max() + 1

# MAKE UP TIME: per ID - time TOTAL_TIME --------------------------------------
df = df.sort_values(["ID","Time"])
sequence_times = pd.DataFrame({'Time':range(TOTAL_TIME)})
unique_ids = df["ID"].unique()
IDs = np.repeat(unique_ids, TOTAL_TIME)
Times = np.tile(np.arange(TOTAL_TIME), len(unique_ids))
template = pd.DataFrame({"ID":IDs, "Time":Times})
new = template.merge(df, how='left',on=['ID','Time'])

# trouble shooting (repeated row Time)
temp = new.groupby("ID").count()['Time']
temp = temp[temp>TOTAL_TIME]
new = new[new['ID'].isin(temp.index.values)==False]
del template
del IDs, Times

# Fill out NA ----------------------------------------------------------
new = new.sort_values(["ID","Time"])
new[['Visits','Comorb','VaxDose','AGE90','GENDER','RACE']] = new.groupby("ID").apply(lambda x: x[['Visits','Comorb','VaxDose','AGE90','GENDER','RACE']].fillna(method='ffill')).reset_index(drop=True)

# ------------------------------------------

new["event"].fillna(value=0,inplace=True)
new["Bivalent"].fillna(value='No vac',inplace=True)
bivalent = {key: index for index, key in enumerate(new["Bivalent"].unique())}
new["Bivalent"] = new["Bivalent"].replace(bivalent)

# Create 'TimefromLastDose'
new["t1"] = new.groupby("ID")["t1"].apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
new["TimefromLastDose"] = new["Time"] - new["t1"]
new["TimefromLastDose"].fillna(value = 100, inplace=True)

# MASK - SeqLen per patient: time to last severe infection (2) or DEATH (dont want to learn shortcut)
last_event = (new[new['censor_prev']==1].groupby("ID")['Time'].min() + 1).to_dict()
[last_event.setdefault(key, TOTAL_TIME) for key in unique_ids]

# To one-hot encoding --------------------------------------------
one_hot_encoded = new[["ID","Time"]]
def one_hot_encoding(df, VarName, addFile):
    depth = len(df[VarName].unique())
    one_hot_encoded_var = tf.one_hot(df[VarName], depth).numpy()

    addFile = np.concatenate((addFile,one_hot_encoded_var),axis=1)
    return addFile

# new.columns
tb1 = one_hot_encoding(new,'GENDER', one_hot_encoded)
tb1 = one_hot_encoding(new,'RACE', tb1)
tb1 = one_hot_encoding(new,'Visits', tb1)
tb1 = one_hot_encoding(new,'Comorb', tb1)
tb1 = one_hot_encoding(new,'Bivalent', tb1)
del one_hot_encoded

tb1 = pd.DataFrame(tb1)
tb1['AGE90'] = new['AGE90'].values
tb1['VaxDose'] = new['VaxDose']
tb1.rename(columns={0:"ID",1:"Time"}, inplace=True)

# dictionaries:
print(visits)      # 5
print(comorbidity) # 4
print(bivalent)    # 3
print(gender)      # 2
print(race)        # 3

assert tb1.isnull().any().any() == False

# Calculate the number of IDs to select for test (15% of total unique IDs)
percent_to_test = 0.2
num_ids = len(new['ID'].unique())
num_to_select = int(num_ids * percent_to_test)

# Set a random seed for reproducibility
np.random.seed(123)

# Randomly select unique IDs,  Create test data and train data based on selected IDs
selected_unique_ids = np.random.choice(new['ID'].unique(), num_to_select, replace=False)
test_tb1 = tb1[tb1['ID'].isin(selected_unique_ids)]
train_tb1 = tb1[~tb1['ID'].isin(selected_unique_ids)]
test_new = new[new['ID'].isin(selected_unique_ids)]
train_new = new[~new['ID'].isin(selected_unique_ids)]
last_event_test = (test_new[test_new['censor_prev']==1].groupby("ID")['Time'].min() + 1).to_dict()
last_event_train = (train_new[train_new['censor_prev']==1].groupby("ID")['Time'].min() + 1).to_dict()
[last_event_test.setdefault(key, TOTAL_TIME) for key in test_new['ID'].unique()]
[last_event_train.setdefault(key, TOTAL_TIME) for key in train_new['ID'].unique()]

#Data for training
sequence_length = TOTAL_TIME
num_isolates = len(train_new['ID'].value_counts())# len(unique_ids)
num_vars = train_tb1.shape[1] - 1 #-ID, NOTE: Time as input
reshaped_input = train_tb1.iloc[:,1:].values.reshape(num_isolates, sequence_length, num_vars)
reshaped_output = train_new['event'].values.reshape(num_isolates, sequence_length)
length = [last_event_train[key] for key in sorted(last_event_train.keys())]
fulldata = {'inputs': reshaped_input, 'outputs': reshaped_output, 'length': length}

import pickle
with open("vaccine_train_lstm.pkl", "wb") as f:
    pickle.dump(fulldata, f)

#Data for testing
sequence_length = TOTAL_TIME
num_isolates = len(test_new['ID'].value_counts())# len(unique_ids)
num_vars = test_tb1.shape[1] - 1 #-ID, NOTE: Time as input
reshaped_input = test_tb1.iloc[:,1:].values.reshape(num_isolates, sequence_length, num_vars)
reshaped_output = test_new['event'].values.reshape(num_isolates, sequence_length)
length = [last_event_test[key] for key in sorted(last_event_test.keys())]
fulldata = {'inputs': reshaped_input, 'outputs': reshaped_output, 'length': length}

with open("vaccine_test_lstm.pkl", "wb") as f:
    pickle.dump(fulldata, f)