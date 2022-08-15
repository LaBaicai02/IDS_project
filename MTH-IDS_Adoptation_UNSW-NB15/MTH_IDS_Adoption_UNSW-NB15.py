#!/usr/bin/env python
# coding: utf-8

# # MTH-IDS Adoption Using UNSW-NB15
# This is an adoption of the method described in the paper entitled "[**MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles**](https://arxiv.org/pdf/2105.13289.pdf)" accepted in IEEE Internet of Things Journal using UNSW-NB15 dataset to test its performance.  

# ## Import libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import os
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance


# ## Read the UNSW-NB15 dataset
# The UNSW-NB15 dataset is publicly available at: https://research.unsw.edu.au/projects/unsw-nb15-dataset  
# This notebook uses a merged version of the dataset, please firstly run Prepare_UNSW-NB15_Dataset.ipynb to automatically download the dataset and do the merge  
# *Note: This project downloads the dataset from Kaggle, which is found to be just a copy of the original dataset*

# In[3]:
starting_time = time.time()


#Read dataset
df = pd.read_csv('data/UNSW-NB15.csv')


# In[4]:


df


# In[5]:


df['attack_cat'].value_counts()


# ### Early Feature Selection (remove useless features in the dataset)
# Refering to the training set included in the files downloaded from Kaggle

# In[6]:


# # Read the downloaded training set
# df_training = pd.read_csv('data/achieve/UNSW_NB15_training-set.csv', index_col='id')
# # Get feature set
# features = df_training.columns.to_list()
# # Adjustments
# # Rename some feature names to match the complete dataset
# rename_dict = {
#     'smean': 'smeansz',
#     'dmean': 'dmeansz',
#     'response_body_len': 'res_bdy_len',
#     'sinpkt': 'sintpkt',
#     'dinpkt': 'dintpkt'
# }
# features = list(map(lambda x: rename_dict[x] if x in rename_dict else x, features))
# # Remove feature 'rate' since it is not in the dataset
# features.remove('rate')
# # Remove feature 'label' to match the original code of MTH-IDS_IoTJ
# features.remove('label')
# # Release Memory
# del df_training

# For convenience
features = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
 'sintpkt', 'dintpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smeansz', 'dmeansz', 'trans_depth',
 'res_bdy_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat']

print(features)


# In[7]:


# Only keep the influencing features
df = df[features]


# ### Preprocessing (normalization and padding values)

# In[8]:


# Z-score normalization
features = df.dtypes[df.dtypes != 'object'].index
df[features] = df[features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# Fill empty values by 0
df = df.fillna(0)


# ### Data sampling
# Due to the space limit of GitHub files and the large size of network traffic data, we sample a small-sized subset for model learning using **k-means cluster sampling**

# In[9]:


temp_labels = df['attack_cat'].value_counts().index.to_list()


# In[10]:


labelencoder = LabelEncoder()
for col_name in df.dtypes[df.dtypes == 'object'].index:
      df[col_name] = labelencoder.fit_transform(df[col_name])


# In[11]:


df['attack_cat'].value_counts()


# In[12]:


label_names = []
temp_series = df['attack_cat'].value_counts().index.to_list()
for i in range(len(temp_labels)):
    label_names.append(temp_labels[temp_series.index(i)])
del temp_labels
del temp_series
label_names


# In[13]:


# retain the minority class instances and sample the majority class instances
df_minor = df[(df['attack_cat']==0)|(df['attack_cat']==1)|(df['attack_cat']==9)|(df['attack_cat']==2)|(df['attack_cat']==10)]
df_major = df.drop(df_minor.index)


# In[14]:


X = df_major.drop(['attack_cat'],axis=1) 
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


# In[15]:


# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)


# In[16]:


klabel=kmeans.labels_
df_major['klabel']=klabel


# In[17]:


df_major['klabel'].value_counts()


# In[18]:


cols = list(df_major)
cols.insert(41, cols.pop(cols.index('attack_cat')))
df_major = df_major.loc[:, cols]


# In[19]:


df_major


# In[20]:


def typicalSampling(group):
    name = group.name
    frac = 0.024
    return group.sample(frac=frac)

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)


# In[21]:


result['attack_cat'].value_counts()


# In[22]:


result


# In[23]:


result = result.drop(['klabel'],axis=1)
result = result.append(df_minor)


# In[24]:


result.to_csv('./data/UNSW-NB15_sample_km.csv',index=0)


# ### split train set and test set

# In[25]:


df=pd.read_csv('./data/UNSW-NB15_sample_km.csv')


# In[26]:


X = df.drop(['attack_cat'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# ## Feature engineering

# In[28]:


# Prepare the result output
output_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Train_time', 'Predict_time_per_record', 'HPO_time'])
output_index = list()


# ### Feature selection by information gain

# In[29]:


from sklearn.feature_selection import mutual_info_classif
importances = mutual_info_classif(X_train, y_train)


# In[30]:


# calculate the sum of importance scores
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])


# In[31]:


# select the important features from top to bottom until the accumulated importance reaches 90%
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2>=0.9:
        break        


# In[32]:


X_fs = df[fs].values


# In[33]:


X_fs.shape


# ### Feature selection by Fast Correlation Based Filter (FCBF)
# 
# The module is imported from the GitHub repo: https://github.com/SantiagoEG/FCBF_module

# In[34]:


from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
fcbf = FCBFK(k = 20)
#fcbf.fit(X_fs, y)


# In[35]:


t1 = time.time()
X_fss = fcbf.fit_transform(X_fs,y)
t2 = time.time()


# In[36]:


# Add to output sheet
result_dict = {
    'Accuracy': np.NaN,
    'Precision': np.NaN,
    'Recall': np.NaN,
    'F1-Score': np.NaN,
    'Train_time': t2-t1,
    'Predict_time_per_record': np.NaN,
    'HPO_time': np.NaN
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('FCBF')


# In[37]:


X_fss.shape


# ### Re-split train & test sets after feature selection

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# In[39]:


X_train.shape


# In[40]:


pd.Series(y_train).value_counts()


# ### SMOTE to solve class-imbalance

# In[41]:


sampling_strategy = dict()
smote_threshold = 1000
for key, value in df['attack_cat'].value_counts().to_dict().items():
    if value < smote_threshold: sampling_strategy[key]=smote_threshold


# In[42]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy=sampling_strategy)


# In[43]:


X_train, y_train = smote.fit_resample(X_train, y_train)


# In[44]:


pd.Series(y_train).value_counts()


# In[45]:


X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)


# ## Machine learning model training

# ### Training four base learners: decision tree, random forest, extra trees, XGBoost

# In[46]:


# The length of the test set for prediction time measurement
len_test = X_test.shape[0]
# Prepare the output dir
output_dir = 'output/MTH-IDS/output-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
img_dir = os.path.join(output_dir, 'img')
os.makedirs(img_dir)
# Prepare the log file
log_file = open(os.path.join(output_dir, 'classification_report-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))), 'w+')


# #### Apply XGBoost

# In[47]:


xg = xgb.XGBClassifier(n_estimators = 10)
t1 = time.time()
xg.fit(X_train,y_train)
t2 = time.time()
xg_score=xg.score(X_test,y_test)
t3 = time.time()
y_predict=xg.predict(X_test)
t4 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('XGBoost (Original)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=1,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'XGBoost_original.pdf'))
# plt.show()


# In[48]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Train_time': t2-t1,
    'Predict_time_per_record': (t4-t3)/len_test,
    'HPO_time': np.NaN
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (Original)')


# #### Hyperparameter optimization (HPO) of XGBoost using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[49]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'learning_rate':  abs(float(params['learning_rate'])),

    }
    clf = xgb.XGBClassifier( **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return {'loss':-score, 'status': STATUS_OK }

space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
}

t1 = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

t2 = time.time()

print("XGBoost: Hyperopt estimated optimum {}".format(best))


# In[50]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'learning_rate':  abs(float(best['learning_rate'])),
}
xg = xgb.XGBClassifier(**params)
t3 = time.time()
xg.fit(X_train,y_train)
t4 = time.time()
xg_score=xg.score(X_test,y_test)
t5 = time.time()
y_predict=xg.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('XGBoost (BO-TPE)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'XGBoost_BO-TPE.pdf'))
# plt.show()


# In[51]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (BO-TPE)')


# In[52]:


xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)


# #### Hyperparameter optimization (HPO) of XGBoost using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[53]:


#XGBoost
import optunity
import optunity.metrics

data= X_combined
labels= y_combined.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'n_estimators': [10, 100],
    'max_depth': [5,50],
    'learning_rate': [0.01, 0.9]
}
# Define the objective function
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_depth=None,learning_rate=None):
    # fit the model
    params = {
        'n_estimators': int(n_estimators), 
        'max_depth': int(max_depth),
        'learning_rate':  abs(float(learning_rate)),
    }
    model = xgb.XGBClassifier( **params)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
    #                                 scoring="accuracy"))
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return optunity.metrics.accuracy(Y_test, predictions)

t1 = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

t2 = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[54]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'learning_rate': abs(float(optimal_configuration['learning_rate']))
}
xg = xgb.XGBClassifier(**params)
t3 = time.time()
xg.fit(X_train,y_train)
t4 = time.time()
xg_score=xg.score(X_test,y_test)
t5 = time.time()
y_predict=xg.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('XGBoost (PSO)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'XGBoost_PSO.pdf'))
# plt.show()


# In[55]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (PSO)')


# #### Hyperparameter optimization (HPO) of XGBoost using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[56]:


#Xgboost
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(10,100),
    'max_depth': range(4,100),
    'learning_rate': [i/100 for i in range(1, 90)]
}
# Set the hyperparameters of GA                 
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'xgboost.XGBClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
t1 = time.time()
ga.fit(X_combined, y_combined)
t2 = time.time()


# In[57]:


# Helper method: convert the values represented by string to its correct type
def type_str(input_str:str):
    # is integer
    if input_str.isdecimal():
        return int(input_str)
    # is float
    elif input_str.isdigit():
        return float(input_str)
    # is string
    elif input_str.startswith('"') and input_str.endswith('"'):
        # remove quotation marks
        return input_str[1: -1]
    else:
        return input_str

# Extract the optimized parameter from the generated pipeline
def get_ga_optimized_parameters(fitted_tpot_obj: TPOTClassifier, classifier_name: str, temp_file_name:str='temp_ga_pipeline.py'):
    # Export the pipeline
    fitted_tpot_obj.export(output_file_name=temp_file_name)
    # Read the optimized pipeline
    with open(temp_file_name) as temp_file:
        lines = temp_file.readlines()
    for line in lines:
        if classifier_name+'(' in line.strip():
            pipeline = line
            break
    # Extract the optimized parameters
    start_index = pipeline.index(classifier_name+'(')
    end_index = pipeline.index(')')
    parameters_str = pipeline[start_index+len(classifier_name)+1: end_index]
    parameters = dict()
    for temp_str in parameters_str.split(sep=','):
        temp_list = temp_str.split('=')
        parameters[temp_list[0].strip()] = type_str(temp_list[1].strip())
    # Delect the temp file
    os.remove(temp_file_name)
    # Return the optimized parameters
    return parameters


# In[58]:


xg = xgb.XGBClassifier(**get_ga_optimized_parameters(ga, 'XGBClassifier'))
t3 = time.time()
xg.fit(X_train,y_train)
t4 = time.time()
xg_score=xg.score(X_test,y_test)
t5 = time.time()
y_predict=xg.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('XGBoost (GA)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'XGBoost_GA.pdf'))
# plt.show()


# In[59]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (GA)')


# #### Apply RF

# In[60]:


rf = RandomForestClassifier(random_state = 0)
t1 = time.time()
rf.fit(X_train,y_train)
t2 = time.time() 
rf_score=rf.score(X_test,y_test)
t3 = time.time()
y_predict=rf.predict(X_test)
t4 = time.time()
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('RF (Original)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'RF_original.pdf'))
# plt.show()


# In[61]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Train_time': t2-t1,
    'Predict_time_per_record': (t4/t3)/len_test,
    'HPO_time': np.NaN
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (Original)')


# #### Hyperparameter optimization (HPO) of random forest using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[62]:


# Hyperparameter optimization of random forest
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Define the objective function
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        "min_samples_split":int(params['min_samples_split']),
        "min_samples_leaf":int(params['min_samples_leaf']),
        "criterion":str(params['criterion'])
    }
    clf = RandomForestClassifier( **params)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)

    return {'loss':-score, 'status': STATUS_OK }
# Define the hyperparameter configuration space
available_criterion = ['gini','entropy']
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    "max_features":hp.quniform('max_features', 1, 20, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion', available_criterion)
}

t1 = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

t2 = time.time()

print("Random Forest: Hyperopt estimated optimum {}".format(best))


# In[63]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
rf_hpo = RandomForestClassifier(**params)
t3 = time.time()
rf_hpo.fit(X_train,y_train)
t4 = time.time()
rf_score=rf_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=rf_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('RF (BO-TPE)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'RF_BO-TPE.pdf'))
# plt.show()


# In[64]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (BO-TPE)')


# In[65]:


rf_train=rf_hpo.predict(X_train)
rf_test=rf_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of random forest using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[66]:


#Random Forest
import optunity
import optunity.metrics

data=X
labels=y.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'n_estimators': [10, 100],
    'max_features': [1, 20],
    'max_depth': [5,50],
    "min_samples_split":[2,11],
    "min_samples_leaf":[1,11],
    "criterion":[0,1]
         }
available_criterion = ['gini', 'entropy']
# Define the objective function
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None,criterion=None):
    # fit the model
    if criterion<0.5:
        cri=available_criterion[0]
    else:
        cri=available_criterion[1]
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_features=int(max_features),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   criterion=cri,
                                  )
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
    #                                 scoring="accuracy"))
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return optunity.metrics.accuracy(Y_test, predictions)

t1 = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

t2 = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[67]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
rf_hpo = RandomForestClassifier(**params)
t3 = time.time()
rf_hpo.fit(X_train,y_train)
t4 = time.time()
rf_score=rf_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=rf_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('RF (PSO)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'RF_PSO.pdf'))
# plt.show()


# In[68]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (PSO)')


# #### Hyperparameter optimization (HPO) of random forest using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[69]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(20,200),
    "max_features":range(1,20),
    'max_depth': range(10,100),
    "min_samples_split":range(2,11),
    "min_samples_leaf":range(1,11),
    "criterion":['gini','entropy']
             }
# Set the hyperparameters of GA                 
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'sklearn.ensemble.RandomForestClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
t1 = time.time()
ga.fit(X_combined, y_combined)
t2 = time.time()


# In[70]:


rf_hpo = RandomForestClassifier(**get_ga_optimized_parameters(ga, 'RandomForestClassifier'))
t3 = time.time()
rf_hpo.fit(X_train,y_train)
t4 = time.time()
rf_score=rf_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=rf_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('RF (GA)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'RF_GA.pdf'))
# plt.show()


# In[71]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (GA)')


# #### Apply DT

# In[72]:


dt = DecisionTreeClassifier(random_state = 0)
t1 = time.time()
dt.fit(X_train,y_train)
t2 = time.time() 
dt_score=dt.score(X_test,y_test)
t3 = time.time()
y_predict=dt.predict(X_test)
t4 = time.time()
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('DT (Original)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'DT_original.pdf'))
# plt.show()


# In[73]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Train_time': t2-t1,
    'Predict_time_per_record': (t4-t3)/len_test,
    'HPO_time': np.NaN
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (Original)')


# #### Hyperparameter optimization (HPO) of decision tree using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[74]:


# Hyperparameter optimization of decision tree
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Define the objective function
def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        "min_samples_split":int(params['min_samples_split']),
        "min_samples_leaf":int(params['min_samples_leaf']),
        "criterion":str(params['criterion'])
    }
    clf = DecisionTreeClassifier( **params)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)

    return {'loss':-score, 'status': STATUS_OK }
# Define the hyperparameter configuration space
available_criterion = ['gini','entropy']
space = {
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    "max_features":hp.quniform('max_features', 1, 20, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion',available_criterion)
}

t1 = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

t2 = time.time()

print("Decision tree: Hyperopt estimated optimum {}".format(best))


# In[75]:


params = {
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
dt_hpo = DecisionTreeClassifier(**params)
t3 = time.time()
dt_hpo.fit(X_train,y_train)
t4 = time.time()
dt_score=dt_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=dt_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('DT (BO-TPE)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'DT_BO-TPE.pdf'))
# plt.show()


# In[76]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (BO-TPE)')


# In[77]:


dt_train=dt_hpo.predict(X_train)
dt_test=dt_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of decision tree using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[78]:


#Random Forest
import optunity
import optunity.metrics

data=X_train
labels=y_train.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'max_features': [1, 20],
    'max_depth': [5,50],
    "min_samples_split":[2,11],
    "min_samples_leaf":[1,11],
    "criterion":[0,1]
}
available_criterion = ['gini', 'entropy']
# Define the objective function
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None,criterion=None):
    # fit the model
    if criterion<0.5:
        cri=available_criterion[0]
    else:
        cri=available_criterion[1]
    model = DecisionTreeClassifier(max_features=int(max_features),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   criterion=cri,
                                  )
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
                                    # scoring="accuracy"))
    # return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return optunity.metrics.accuracy(Y_test, predictions)

t1 = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

t2 = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[79]:


params = {
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
dt_hpo = DecisionTreeClassifier(**params)
t3 = time.time()
dt_hpo.fit(X_train,y_train)
t4 = time.time()
dt_score=dt_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=dt_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('DT (PSO)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'DT_PSO.pdf'))
# plt.show()


# In[80]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (PSO)')


# #### Hyperparameter optimization (HPO) of decision tree using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[81]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    "max_features":range(1,20),
    'max_depth': range(10,100),
    "min_samples_split":range(2,11),
    "min_samples_leaf":range(1,11),
    "criterion":['gini','entropy']
}
# Set the hyperparameters of GA                 
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'sklearn.tree.DecisionTreeClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
t1 = time.time()
ga.fit(X_combined, y_combined)
t2 = time.time()


# In[82]:


dt_hpo = DecisionTreeClassifier(**get_ga_optimized_parameters(ga, 'DecisionTreeClassifier'))
t3 = time.time()
dt_hpo.fit(X_train,y_train)
t4 = time.time()
dt_score=dt_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=dt_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('DT (GA)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'DT_GA.pdf'))
# plt.show()


# In[83]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (GA)')


# #### Apply ET

# In[84]:


et = ExtraTreesClassifier(random_state = 0)
t1 = time.time()
et.fit(X_train,y_train)
t2 = time.time() 
et_score=et.score(X_test,y_test)
t3 = time.time()
y_predict=et.predict(X_test)
t4 = time.time()
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('ET (Original)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'ET_original.pdf'))
# plt.show()


# In[85]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Train_time': t2-t1,
    'Predict_time_per_record': (t4-t3)/len_test,
    'HPO_time': np.NaN
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (Original)')


# #### Hyperparameter optimization (HPO) of extra trees using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[86]:


# Hyperparameter optimization of extra trees
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Define the objective function
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        "min_samples_split":int(params['min_samples_split']),
        "min_samples_leaf":int(params['min_samples_leaf']),
        "criterion":str(params['criterion'])
    }
    clf = ExtraTreesClassifier( **params)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)

    return {'loss':-score, 'status': STATUS_OK }
# Define the hyperparameter configuration space
available_criterion = ['gini','entropy']
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    "max_features":hp.quniform('max_features', 1, 20, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion',available_criterion)
}

t1 = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

t2 = time.time()

print("Random Forest: Hyperopt estimated optimum {}".format(best))


# In[87]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
et_hpo = ExtraTreesClassifier(**params)
t3 = time.time()
et_hpo.fit(X_train,y_train) 
t4 = time.time()
et_score=et_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=et_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('ET (BO-TPE)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'ET_BO-TPE.pdf'))
# plt.show()


# In[88]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (BO-TPE)')


# In[89]:


et_train=et_hpo.predict(X_train)
et_test=et_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of extra trees using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[90]:


#Random Forest
import optunity
import optunity.metrics

data=X_train
labels=y_train.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'n_estimators': [10, 200],
    'max_features': [1, 20],
    'max_depth': [5,50],
    "min_samples_split":[2,11],
    "min_samples_leaf":[1,11],
    "criterion":[0,1]
}
available_criterion = ['gini', 'entropy']
# Define the objective function
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,n_estimators=None,max_features=None,max_depth=None,min_samples_split=None,min_samples_leaf=None,criterion=None):
    # fit the model
    if criterion<0.5:
        cri=available_criterion[0]
    else:
        cri=available_criterion[1]
    model = ExtraTreesClassifier(n_estimators=int(n_estimators),
                                   max_features=int(max_features),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   criterion=cri,
                                  )
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
                                    # scoring="accuracy"))
    # return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return optunity.metrics.accuracy(Y_test, predictions)

t1 = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

t2 = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[91]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
et_hpo = ExtraTreesClassifier(**params)
t3 = time.time()
et_hpo.fit(X_train,y_train)
t4 = time.time() 
et_score=et_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=et_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('ET (PSO)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'ET_PSO.pdf'))
# plt.show()


# In[92]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (PSO)')


# #### Hyperparameter optimization (HPO) of extra trees using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[93]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(20,200),
    "max_features":range(1,20),
    'max_depth': range(10,100),
    "min_samples_split":range(2,11),
    "min_samples_leaf":range(1,11),
    "criterion":['gini','entropy']
             }
# Set the hyperparameters of GA                 
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'sklearn.ensemble.ExtraTreesClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
t1 = time.time()
ga.fit(X_combined, y_combined)
t2 = time.time()


# In[94]:


et_hpo = ExtraTreesClassifier(**get_ga_optimized_parameters(ga, 'ExtraTreesClassifier'))
t3 = time.time()
et_hpo.fit(X_train,y_train)
t4 = time.time() 
et_score=et_hpo.score(X_test,y_test)
t5 = time.time()
y_predict=et_hpo.predict(X_test)
t6 = time.time()
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('ET (GA)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'ET_GA.pdf'))
# plt.show()


# In[95]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (GA)')


# ### Apply stacking

# In[96]:


base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
    'RandomForest': rf_train.ravel(),
    'ExtraTrees': et_train.ravel(),
    'XgBoost': xg_train.ravel(),
    })
base_predictions_train.head(5)


# In[97]:


dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)


# In[98]:


dt_train.shape


# In[99]:


x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)


# In[100]:


t1 = time.time()
stk = xgb.XGBClassifier().fit(x_train, y_train)
t2 = time.time()
y_predict=stk.predict(x_test)
t3 = time.time()
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('Stacking (Original)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'Stacking_original.pdf'))
# plt.show()


# In[101]:


# Add to output sheet
result_dict = {
    'Accuracy': stk_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Train_time': t2-t1,
    'Predict_time_per_record': (t3-t2)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (Original)')


# #### Hyperparameter optimization (HPO) of the stacking ensemble model (XGBoost) using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[102]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'learning_rate':  abs(float(params['learning_rate'])),

    }
    clf = xgb.XGBClassifier( **params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)

    return {'loss':-score, 'status': STATUS_OK }

space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
}

t1 = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

t2 = time.time()

print("XGBoost: Hyperopt estimated optimum {}".format(best))


# In[103]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'learning_rate':  abs(float(best['learning_rate'])),
}
xg = xgb.XGBClassifier(**params)
t3 = time.time()
xg.fit(x_train,y_train)
t4 = time.time()
xg_score=xg.score(x_test,y_test)
t5 = time.time()
y_predict=xg.predict(x_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('Stacking (BO-TPE)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'Stacking_BO-TPE.pdf'))
# plt.show()


# In[104]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (BO-TPE)')


# #### Hyperparameter optimization (HPO) of stacking ensemble model (XGBoost) using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[105]:


#XGBoost
import optunity
import optunity.metrics

data= X_combined
labels= y_combined.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'n_estimators': [10, 100],
    'max_depth': [5,50],
    'learning_rate': [0.01, 0.9]
}
# Define the objective function
@optunity.cross_validated(x=data, y=labels, num_folds=3)
def performance(x_train, y_train, x_test, y_test,n_estimators=None, max_depth=None,learning_rate=None):
    # fit the model
    params = {
        'n_estimators': int(n_estimators), 
        'max_depth': int(max_depth),
        'learning_rate':  abs(float(learning_rate)),
    }
    model = xgb.XGBClassifier( **params)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # scores=np.mean(cross_val_score(model, X, y, cv=3, n_jobs=-1,
    #                                 scoring="accuracy"))
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return optunity.metrics.accuracy(Y_test, predictions)

t1 = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

t2 = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[106]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'learning_rate': abs(float(optimal_configuration['learning_rate']))
}
xg = xgb.XGBClassifier(**params)
t3 = time.time()
xg.fit(x_train,y_train)
t4 = time.time()
xg_score=xg.score(x_test,y_test)
t5 = time.time()
y_predict=xg.predict(x_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('Stacking (PSO)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'Stacking_PSO.pdf'))
# plt.show()


# In[107]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (PSO)')


# #### Hyperparameter optimization (HPO) of stacking ensemble model (XGBoost) using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[108]:


#XGBoost
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(10,100),
    'max_depth': range(4,100),
    'learning_rate': [i/100 for i in range(1, 90)]
}
# Set the hyperparameters of GA                 
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'xgboost.XGBClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
t1 = time.time()
ga.fit(X_combined, y_combined)
t2 = time.time()


# In[109]:


xg = xgb.XGBClassifier(**get_ga_optimized_parameters(ga, 'XGBClassifier'))
t3 = time.time()
xg.fit(x_train,y_train)
t4 = time.time()
xg_score=xg.score(x_test,y_test)
t5 = time.time()
y_predict=xg.predict(x_test)
t6 = time.time()
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
report_str = classification_report(y_true,y_predict,target_names=label_names); log_file.write('******{}******\n'.format('Stacking (GA)')+report_str+'\n'); print(report_str)
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(18,14))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
ax.set_xticklabels(label_names)
ax.set_yticklabels(list(reversed(label_names)))
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig(os.path.join(img_dir, 'Stacking_GA.pdf'))
# plt.show()


# In[110]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'HPO_time': t2-t1,
    'Train_time': t4-t3,
    'Predict_time_per_record': (t6-t5)/len_test
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (GA)')


# In[111]:


# Rename the index
output_df.index = output_index
# Save the result to file
output_df.to_excel(os.path.join(output_dir, 'result-{}.xlsx'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))))
# Close the logging file
log_file.close()

# Online GPU renting platform specification
# WeChat Message
import requests
resp = requests.get(
    "https://www.autodl.com/api/v1/wechat/message/push?token={token}&title={title}&name={name}&content={content}".format(
        token="",
        title="Running Completed",
        name="MTH_IDS_Adoption_UNSW-NB15.py",
        content="Time Used: {}".format(time.time()-starting_time))
)
print(resp.content.decode())


# ## Anomaly-based IDS

# ### Generate the port-scan datasets for unknown attack detection

# In[112]:


# df=pd.read_csv('./data/UNSW-NB15_sample_km.csv')


# In[113]:


# df['attack_cat'].value_counts()


# In[114]:


# df1 = df[df['attack_cat'] != 9]
# df1['attack_cat'][df1['attack_cat'] != 7] = 1
# df1.to_csv('./data/UNSW-NB15_sample_km_without_shellcode.csv',index=0)


# In[115]:


# df2 = df[df['attack_cat'] == 9]
# df2['attack_cat'][df2['attack_cat'] == 9] = 1
# df2.to_csv('./data/UNSW-NB15_sample_km_shellcode.csv',index=0)


# ### Read the generated datasets for unknown attack detection

# In[116]:


# df1 = pd.read_csv('./data/UNSW-NB15_sample_km_without_shellcode.csv')
# df2 = pd.read_csv('./data/UNSW-NB15_sample_km_shellcode.csv')


# In[117]:


# features = df1.drop(['attack_cat'],axis=1).dtypes[df1.dtypes != 'object'].index
# df1[features] = df1[features].apply(
#     lambda x: (x - x.mean()) / (x.std()))
# df2[features] = df2[features].apply(
#     lambda x: (x - x.mean()) / (x.std()))
# df1 = df1.fillna(0)
# df2 = df2.fillna(0)


# In[118]:


# df1['attack_cat'].value_counts()


# In[119]:


# df2['attack_cat'].value_counts()


# In[120]:


# df2p=df1[df1['attack_cat']==7]
# df2pp=df2p.sample(n=None, frac=1511/53253, replace=False, weights=None, random_state=None, axis=0)
# df2=pd.concat([df2, df2pp])


# In[121]:


# df2['attack_cat'].value_counts()


# In[122]:


# df = df1.append(df2)


# In[123]:


# X = df.drop(['attack_cat'],axis=1).values
# y = df.iloc[:, -1].values.reshape(-1,1)
# y=np.ravel(y)
# pd.Series(y).value_counts()


# ### Feature engineering (IG, FCBF, and KPCA)

# #### Feature selection by information gain (IG)

# In[124]:


# from sklearn.feature_selection import mutual_info_classif
# importances = mutual_info_classif(X, y)


# In[125]:


# # calculate the sum of importance scores
# f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
# Sum = 0
# fs = []
# for i in range(0, len(f_list)):
#     Sum = Sum + f_list[i][0]
#     fs.append(f_list[i][1])


# In[126]:


# # select the important features from top to bottom until the accumulated importance reaches 90%
# f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
# Sum2 = 0
# fs = []
# for i in range(0, len(f_list2)):
#     Sum2 = Sum2 + f_list2[i][0]
#     fs.append(f_list2[i][1])
#     if Sum2>=0.9:
#         break        


# In[127]:


# X_fs = df[fs].values


# In[128]:


# X_fs.shape


# In[129]:


# X_fs


# #### Feature selection by Fast Correlation Based Filter (FCBF)
# 
# The module is imported from the GitHub repo: https://github.com/SantiagoEG/FCBF_module

# In[130]:


# from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
# fcbf = FCBFK(k = 20)
# #fcbf.fit(X_fs, y)


# In[131]:


# X_fss = fcbf.fit_transform(X_fs,y)


# In[132]:


# X_fss.shape


# In[133]:


# X_fss


# ####  kernel principal component analysis (KPCA)

# In[134]:


# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components = 10, kernel = 'rbf')
# kpca.fit(X_fss, y)
# X_kpca = kpca.transform(X_fss)

# # from sklearn.decomposition import PCA
# # kpca = PCA(n_components = 10)
# # kpca.fit(X_fss, y)
# # X_kpca = kpca.transform(X_fss)


# ### Train-test split after feature selection

# In[135]:


# X_train = X_kpca[:len(df1)]
# y_train = y[:len(df1)]
# X_test = X_kpca[len(df1):]
# y_test = y[len(df1):]


# ### Solve class-imbalance by SMOTE

# In[136]:


# pd.Series(y_train).value_counts()


# In[137]:


# from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={1:18225})
# X_train, y_train = smote.fit_sample(X_train, y_train)


# In[138]:


# pd.Series(y_train).value_counts()


# In[139]:


# pd.Series(y_test).value_counts()


# ### Apply the cluster labeling (CL) k-means method

# In[140]:


# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN,MeanShift
# from sklearn.cluster import SpectralClustering,AgglomerativeClustering,AffinityPropagation,Birch,MiniBatchKMeans,MeanShift 
# from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
# from sklearn.metrics import classification_report
# from sklearn import metrics


# In[141]:


# def CL_kmeans(X_train, X_test, y_train, y_test,n,b=100):
#     km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b)
#     result = km_cluster.fit_predict(X_train)
#     result2 = km_cluster.predict(X_test)

#     count=0
#     a=np.zeros(n)
#     b=np.zeros(n)
#     for v in range(0,n):
#         for i in range(0,len(y_train)):
#             if result[i]==v:
#                 if y_train[i]==1:
#                     a[v]=a[v]+1
#                 else:
#                     b[v]=b[v]+1
#     list1=[]
#     list2=[]
#     for v in range(0,n):
#         if a[v]<=b[v]:
#             list1.append(v)
#         else: 
#             list2.append(v)
#     for v in range(0,len(y_test)):
#         if result2[v] in list1:
#             result2[v]=0
#         elif result2[v] in list2:
#             result2[v]=1
#         else:
#             print("-1")
#     print(classification_report(y_test, result2))
#     cm=confusion_matrix(y_test,result2)
#     acc=metrics.accuracy_score(y_test,result2)
#     print(str(acc))
#     print(cm)


# In[142]:


# CL_kmeans(X_train, X_test, y_train, y_test, 8)


# ### Hyperparameter optimization of CL-k-means
# Tune "k"

# In[143]:


# #Hyperparameter optimization by BO-GP
# from skopt.space import Real, Integer
# from skopt.utils import use_named_args
# from sklearn import metrics

# space  = [Integer(2, 50, name='n_clusters')]
# @use_named_args(space)
# def objective(**params):
#     km_cluster = MiniBatchKMeans(batch_size=100, **params)
#     n=params['n_clusters']
    
#     result = km_cluster.fit_predict(X_train)
#     result2 = km_cluster.predict(X_test)

#     count=0
#     a=np.zeros(n)
#     b=np.zeros(n)
#     for v in range(0,n):
#         for i in range(0,len(y_train)):
#             if result[i]==v:
#                 if y_train[i]==1:
#                     a[v]=a[v]+1
#                 else:
#                     b[v]=b[v]+1
#     list1=[]
#     list2=[]
#     for v in range(0,n):
#         if a[v]<=b[v]:
#             list1.append(v)
#         else: 
#             list2.append(v)
#     for v in range(0,len(y_test)):
#         if result2[v] in list1:
#             result2[v]=0
#         elif result2[v] in list2:
#             result2[v]=1
#         else:
#             print("-1")
#     cm=metrics.accuracy_score(y_test,result2)
#     print(str(n)+" "+str(cm))
#     return (1-cm)
# from skopt import gp_minimize
# import time
# t1=time.time()
# res_gp = gp_minimize(objective, space, n_calls=20, random_state=0)
# t2=time.time()
# print(t2-t1)
# print("Best score=%.4f" % (1-res_gp.fun))
# print("""Best parameters: n_clusters=%d""" % (res_gp.x[0]))


# In[144]:


# #Hyperparameter optimization by BO-TPE
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.cluster import MiniBatchKMeans
# from sklearn import metrics

# def objective(params):
#     params = {
#         'n_clusters': int(params['n_clusters']), 
#     }
#     km_cluster = MiniBatchKMeans(batch_size=100, **params)
#     n=params['n_clusters']
    
#     result = km_cluster.fit_predict(X_train)
#     result2 = km_cluster.predict(X_test)

#     count=0
#     a=np.zeros(n)
#     b=np.zeros(n)
#     for v in range(0,n):
#         for i in range(0,len(y_train)):
#             if result[i]==v:
#                 if y_train[i]==1:
#                     a[v]=a[v]+1
#                 else:
#                     b[v]=b[v]+1
#     list1=[]
#     list2=[]
#     for v in range(0,n):
#         if a[v]<=b[v]:
#             list1.append(v)
#         else: 
#             list2.append(v)
#     for v in range(0,len(y_test)):
#         if result2[v] in list1:
#             result2[v]=0
#         elif result2[v] in list2:
#             result2[v]=1
#         else:
#             print("-1")
#     score=metrics.accuracy_score(y_test,result2)
#     print(str(params['n_clusters'])+" "+str(score))
#     return {'loss':1-score, 'status': STATUS_OK }
# space = {
#     'n_clusters': hp.quniform('n_clusters', 2, 50, 1),
# }

# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=20)
# print("Random Forest: Hyperopt estimated optimum {}".format(best))


# In[145]:


# CL_kmeans(X_train, X_test, y_train, y_test, 16)


# ### Apply the CL-k-means model with biased classifiers

# In[146]:


# # needs to work on the entire dataset to generate sufficient training samples for biased classifiers
# def Anomaly_IDS(X_train, X_test, y_train, y_test,n,b=100):
#     # CL-kmeans
#     km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b)
#     result = km_cluster.fit_predict(X_train)
#     result2 = km_cluster.predict(X_test)

#     count=0
#     a=np.zeros(n)
#     b=np.zeros(n)
#     for v in range(0,n):
#         for i in range(0,len(y_train)):
#             if result[i]==v:
#                 if y_train[i]==1:
#                     a[v]=a[v]+1
#                 else:
#                     b[v]=b[v]+1
#     list1=[]
#     list2=[]
#     for v in range(0,n):
#         if a[v]<=b[v]:
#             list1.append(v)
#         else: 
#             list2.append(v)
#     for v in range(0,len(y_test)):
#         if result2[v] in list1:
#             result2[v]=0
#         elif result2[v] in list2:
#             result2[v]=1
#         else:
#             print("-1")
#     print(classification_report(y_test, result2))
#     cm=confusion_matrix(y_test,result2)
#     acc=metrics.accuracy_score(y_test,result2)
#     print(str(acc))
#     print(cm)
    
#     #Biased classifier construction
#     count=0
#     print(len(y))
#     a=np.zeros(n)
#     b=np.zeros(n)
#     FNL=[]
#     FPL=[]
#     for v in range(0,n):
#         al=[]
#         bl=[]
#         for i in range(0,len(y)):   
#             if result[i]==v:        
#                 if y[i]==1:        #label 1
#                     a[v]=a[v]+1
#                     al.append(i)
#                 else:             #label 0
#                     b[v]=b[v]+1
#                     bl.append(i)
#         if a[v]<=b[v]:
#             FNL.extend(al)
#         else:
#             FPL.extend(bl)
#         #print(str(v)+"="+str(a[v]/(a[v]+b[v])))
        
#     dffp=df.iloc[FPL, :]
#     dffn=df.iloc[FNL, :]
#     dfva0=df[df['Label']==0]
#     dfva1=df[df['Label']==1]
    
#     dffpp=dfva1.sample(n=None, frac=len(FPL)/dfva1.shape[0], replace=False, weights=None, random_state=None, axis=0)
#     dffnp=dfva0.sample(n=None, frac=len(FNL)/dfva0.shape[0], replace=False, weights=None, random_state=None, axis=0)
    
#     dffp_f=pd.concat([dffp, dffpp])
#     dffn_f=pd.concat([dffn, dffnp])
    
#     Xp = dffp_f.drop(['Label'],axis=1)  
#     yp = dffp_f.iloc[:, -1].values.reshape(-1,1)
#     yp=np.ravel(yp)

#     Xn = dffn_f.drop(['Label'],axis=1)  
#     yn = dffn_f.iloc[:, -1].values.reshape(-1,1)
#     yn=np.ravel(yn)
    
#     rfp = RandomForestClassifier(random_state = 0)
#     rfp.fit(Xp,yp)
#     rfn = RandomForestClassifier(random_state = 0)
#     rfn.fit(Xn,yn)

#     dffnn_f=pd.concat([dffn, dffnp])
    
#     Xnn = dffn_f.drop(['Label'],axis=1)  
#     ynn = dffn_f.iloc[:, -1].values.reshape(-1,1)
#     ynn=np.ravel(ynn)

#     rfnn = RandomForestClassifier(random_state = 0)
#     rfnn.fit(Xnn,ynn)

#     X2p = df2.drop(['Label'],axis=1) 
#     y2p = df2.iloc[:, -1].values.reshape(-1,1)
#     y2p=np.ravel(y2p)

#     result2 = km_cluster.predict(X2p)

#     count=0
#     a=np.zeros(n)
#     b=np.zeros(n)
#     for v in range(0,n):
#         for i in range(0,len(y)):
#             if result[i]==v:
#                 if y[i]==1:
#                     a[v]=a[v]+1
#                 else:
#                     b[v]=b[v]+1
#     list1=[]
#     list2=[]
#     l1=[]
#     l0=[]
#     for v in range(0,n):
#         if a[v]<=b[v]:
#             list1.append(v)
#         else: 
#             list2.append(v)
#     for v in range(0,len(y2p)):
#         if result2[v] in list1:
#             result2[v]=0
#             l0.append(v)
#         elif result2[v] in list2:
#             result2[v]=1
#             l1.append(v)
#         else:
#             print("-1")
#     print(classification_report(y2p, result2))
#     cm=confusion_matrix(y2p,result2)
#     print(cm)


# More details are in the paper
