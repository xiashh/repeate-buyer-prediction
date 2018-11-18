import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

user_log_path = 'dataset/user_log_format1.csv'
user_log_dataset = pd.read_csv(user_log_path,low_memory=False)
print(user_log_dataset.head())


f1 = user_log_dataset[['user_id']].copy()
f1['user_handle_action_count'] = 1
f1 = f1.groupby(['user_id']).agg('sum').reset_index()
f1.head()


user_info_path = 'dataset/user_info_format1.csv'
user_info_dataset = pd.read_csv(user_info_path,low_memory=False)
print(user_info_dataset.head())



f2 = user_info_dataset[['user_id','gender']].copy()
f2 = f2.fillna(2.0)
gender_hot = label_binarize(np.array(f2.gender), classes=[0, 1, 2])
gender_hot_df = pd.DataFrame(gender_hot,columns=['gender_hot_0','gender_hot_1','gender_hot_2'])
f2['gender_hot_0'] = gender_hot_df.gender_hot_0
f2['gender_hot_1'] = gender_hot_df.gender_hot_1
f2['gender_hot_2'] = gender_hot_df.gender_hot_2
f2 = f2.drop(columns=['gender'])
f2.head()

f3 = user_log_dataset[['user_id','action_type']].copy()
action_type_hot = label_binarize(np.array(f3.action_type), classes=[0, 1, 2, 3])
action_type_hot_df = pd.DataFrame(action_type_hot,columns=['action_type_hot_0','action_type_hot_1','action_type_hot_2','action_type_hot_3'])
f3['action_type_hot_0'] = action_type_hot_df.action_type_hot_0
f3['action_type_hot_1'] = action_type_hot_df.action_type_hot_1
f3['action_type_hot_2'] = action_type_hot_df.action_type_hot_2
f3['action_type_hot_3'] = action_type_hot_df.action_type_hot_3
f4 = f3.copy()
f4 = f4.groupby('user_id').agg('sum').reset_index()
f4 = f4.rename(index=str,columns={"action_type_hot_0":"action_type_hot_0_count","action_type_hot_1":"action_type_hot_1_count","action_type_hot_2":"action_type_hot_2_count","action_type_hot_3":"action_type_hot_3_count",})
f4 = f4.drop(columns=['action_type'])
f4.head()

																																																																																																																																																																																																																																																																																																			


fi1 = user_log_dataset[['user_id','action_type']].copy()
action_type_hot = label_binarize(np.array(fi1.action_type), classes=[0, 1, 2, 3])
action_type_hot_df = pd.DataFrame(action_type_hot,columns=['action_type_hot_0','action_type_hot_1','action_type_hot_2','action_type_hot_3'])
fi1['action_type_hot_0'] = action_type_hot_df.action_type_hot_0
fi1['action_type_hot_1'] = action_type_hot_df.action_type_hot_1
fi1['action_type_hot_2'] = action_type_hot_df.action_type_hot_2
fi1['action_type_hot_3'] = action_type_hot_df.action_type_hot_3
fi2 = fi1.copy()
fi2 = fi2.groupby('user_id').agg('sum').reset_index()
fi2 = fi2.rename(index=str,columns={"action_type_hot_0":"user_action_type_hot_0_count","action_type_hot_1":"user_action_type_hot_1_count","action_type_hot_2":"user_action_type_hot_2_count","action_type_hot_3":"user_action_type_hot_3_count",})
fi2 = fi2.drop(columns=['action_type'])
fi2.head()

fi3 = user_log_dataset[['merchant_id','action_type']].copy()
action_type_hot = label_binarize(np.array(fi3.action_type), classes=[0, 1, 2, 3])
action_type_hot_df = pd.DataFrame(action_type_hot,columns=['action_type_hot_0','action_type_hot_1','action_type_hot_2','action_type_hot_3'])
fi3['action_type_hot_0'] = action_type_hot_df.action_type_hot_0
fi3['action_type_hot_1'] = action_type_hot_df.action_type_hot_1
fi3['action_type_hot_2'] = action_type_hot_df.action_type_hot_2
fi3['action_type_hot_3'] = action_type_hot_df.action_type_hot_3
fi4 = fi3.copy()
fi4 = fi4.groupby('merchant_id').agg('sum').reset_index()
fi4 = fi4.rename(index=str,columns={"action_type_hot_0":"merchant_action_type_hot_0_count","action_type_hot_1":"merchant_action_type_hot_1_count","action_type_hot_2":"merchant_action_type_hot_2_count","action_type_hot_3":"merchant_action_type_hot_3_count",})
fi4 = fi4.drop(columns=['action_type'])
fi4.head()

fi5 = user_log_dataset[['user_id','action_type','seller_id']].copy()
action_type_hot = label_binarize(np.array(fi5.action_type), classes=[0, 1, 2, 3])
action_type_hot_df = pd.DataFrame(action_type_hot,columns=['action_type_hot_0','action_type_hot_1','action_type_hot_2','action_type_hot_3'])
fi5['action_type_hot_0'] = action_type_hot_df.action_type_hot_0
fi5['action_type_hot_1'] = action_type_hot_df.action_type_hot_1
fi5['action_type_hot_2'] = action_type_hot_df.action_type_hot_2
fi5['action_type_hot_3'] = action_type_hot_df.action_type_hot_3
fi6 = fi5.copy()
fi6 = fi6.groupby('user_id','mers').agg('sum').reset_index()
fi6 = fi6.rename(index=str,columns={"action_type_hot_0":"user_merchant_action_type_hot_0_count","action_type_hot_1":"user_merchant_action_type_hot_1_count","action_type_hot_2":"user_merchant_action_type_hot_2_count","action_type_hot_3":"user_merchant_action_type_hot_3_count",})
fi6 = fi6.drop(columns=['action_type'])
fi6.head()


u1 = pd.merge(f1,f2,on='user_id')
u2= pd.merge(u1,f4,on='user_id')
uf1=pd.merge(u2,fi2,on='user_id')
uf2=pd.merge(uf1,fi4,on='user_id')
uf3=pd.merge(uf2,fi6,on='user_id')

train_path='dataset/train_format1.csv'
train_dataset=pd.read_csv(train_path,low_memory=False)

total_train_dataset=pd.merge(uf3,train_dataset,on=['user_id','merchant_id'])

total_train_dataset.dropna(how='any')
total_train_dataset.shape
total_train_dataset.to_csv('dataset/total_train_dataset.csv')
dataset = total_train_dataset.drop(columns=['user_id','merchant_id'])
dataset.columns
X = dataset.iloc[:,0:-1]
Y = dataset.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
gbm = xgb.XGBClassifier(silent=1, max_depth=10,
                    n_estimators=1000, learning_rate=0.05)
gbm.fit(X_train, Y_train)
predictions = gbm.predict(X_test)

submission = pd.DataFrame({'user_id': tests['user_id'],
                            'merchant_id': predictions})
print(submission)
submission.to_csv("submission.csv",index=False)

