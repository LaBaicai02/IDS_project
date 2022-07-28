#!/usr/bin/env python
# coding: utf-8

# # MTH of KDDCUP99 Dataset 
# 

# 1. 字符串feature用onehot编码
# 2. 没有分major和minor
# 3. SMOTE把小于1000的设为1000
# 4. 不用手动删只有一个样本的聚类了
# 5. 加了GA，PSO

# ## Import libraries

# In[65]:


import warnings
warnings.filterwarnings("ignore")
import time
import os
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


# ## Read the  dataset

# In[67]:


#Read dataset
df = pd.read_csv('./data/kddcup99.csv')


# In[68]:


df


# In[69]:


df.label.value_counts()


# ### Preprocessing

# In[70]:


# 用onehot处理分类数据
s = (df.dtypes=='object')
object_cols = list(s[s].index)
object_cols #看看哪些列不是数值的


# In[71]:


dfnew = df[['protocol_type', 'service', 'flag']] #3种取值
dfnew.protocol_type.value_counts()


# In[72]:


dfnew.service.value_counts() #66种


# In[73]:


dfnew.flag.value_counts() #11种。 总共80种，被onehot拆分成80个维度,等下编码完拼接起来总共（42+80-3）列


# In[75]:


from sklearn.preprocessing import OneHotEncoder
#进行onehot编码
oh_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
encoded_data = oh_encoder.fit_transform(pd.DataFrame(dfnew))
dfobject = pd.DataFrame(encoded_data) #编码后是array，要先变回dataframe或者series，才能和原来的df拼在一起
dfobject.to_csv('./data/dfobject.csv',index=0)
dfobject


# In[77]:


#还原下名字, 编码后表头是int，必须统一成str，不然后面报错。因为是一个feture分裂成n个列，就取名 原名+序号 吧
sub1 = dfobject.iloc[0:0, 0:3].rename(lambda x:'protocol_type'+str(x+1),axis=1)
sub2 = dfobject.iloc[0:0,3:69].rename(lambda x:'service'+str(x-2),axis=1)
sub3 = dfobject.iloc[0:0,69:80].rename(lambda x:'flag'+str(x-68),axis=1)
sub = pd.concat([sub1,sub2],axis=1)
sub = list(pd.concat([sub,sub3],axis=1))
dfobject = pd.read_csv('./data/dfobject.csv',header=None).drop(0,axis=0) #这里感觉好蠢啊，先存再取哈哈哈哈，我不知道怎么删除旧表头...一开始想用.rename()整体改名的, 结果这个函数里写if一直报错
dfobject.columns = sub
dfobject.index -= 1 #还原索引
dfobject


# In[78]:


#连接主表 42+80-3=119列
df = df.drop(['protocol_type', 'service', 'flag'], axis=1) 
df = pd.concat([dfobject,df],axis=1)
df


# In[79]:


# Z-score normalization
features = df.dtypes[df.dtypes != 'object'].index
df[features] = df[features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# Fill empty values by 0
df = df.fillna(0)


# In[80]:


df


# ### Data sampling
# Due to the space limit of GitHub files and the large size of network traffic data, we sample a small-sized subset for model learning using **k-means cluster sampling**

# In[81]:


labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])


# In[82]:


df.label.value_counts()


# In[83]:


X = df.drop(['label'],axis=1) 
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


# In[84]:


# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)


# In[85]:


klabel=kmeans.labels_
df['klabel']=klabel


# In[86]:


df['klabel'].value_counts()


# In[87]:


cols = list(df)
cols.insert(120, cols.pop(cols.index('label')))
df = df.loc[:, cols]


# In[88]:


df #119列+klabel，120列


# In[89]:


def typicalSampling(group):
    name = group.name
    frac = 0.1 #数据比较少多取了一点
    return group.sample(frac=frac)

result = df.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)


# In[90]:


showValues = result['label'].value_counts()
showValues


# In[91]:


# 筛选并删除只有一个的聚类
delete_values = np.where(showValues==1,showValues.index,-1)
delete_values = np.unique(delete_values).tolist()
del delete_values[0]
print(delete_values)


# In[92]:


result = result[~result['label'].isin(delete_values)]
result['label'].value_counts()


# In[ ]:


result.iloc[:,-1] = labelencoder.fit_transform(result.iloc[:, -1])
result['label'].value_counts()


# In[94]:


result = result.drop(['klabel'],axis=1)


# In[96]:


result.to_csv('./data/cup99_sample_km.csv',index=0)


# ### split train set and test set

# In[97]:


df=pd.read_csv('./data/cup99_sample_km.csv')


# In[98]:


X = df.drop(['label'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


# In[99]:


result['label'].value_counts()


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# ## Feature engineering

# ### Feature selection by information gain

# In[101]:


# Prepare the result output
output_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time'])
output_index = list()


# In[102]:


from sklearn.feature_selection import mutual_info_classif
importances = mutual_info_classif(X_train, y_train)


# In[103]:


#将importance中的数据四舍五入到第四位小数， 压缩打包 reverse=True降序排序
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
#f_list[i][0] 数值 f_list[i][1] 列名 
for i in range(0, len(f_list)): 
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])


# In[104]:


# select the important features from top to bottom until the accumulated importance reaches 90%
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2>=0.5:
        break        


# In[105]:


X_fs = df[fs].values


# In[106]:


X_fs.shape


# ### Feature selection by Fast Correlation Based Filter (FCBF)

# In[121]:


from FCBF_Module import FCBF, FCBFK, FCBFiP, get_i
fcbf = FCBFK(k = 20)
#fcbf.fit(X_fs, y)


# In[122]:


start_time = time.time()
X_fss = fcbf.fit_transform(X_fs,y)
end_time = time.time()


# In[123]:


# Add to output sheet
result_dict = {
    'Accuracy': np.NaN,
    'Precision': np.NaN,
    'Recall': np.NaN,
    'F1-Score': np.NaN,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('FCBF')


# In[124]:


X_fss = fcbf.fit_transform(X_fs,y)


# In[125]:


X_fss.shape


# ### Re-split train & test sets after feature selection

# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# In[127]:


X_train.shape


# In[128]:


minority = pd.Series(y_train).value_counts()
minority


# ### SMOTE to solve class-imbalance

# In[129]:


# 把不足1000个的聚类平衡成1000
smote_values = np.where(minority<1000,minority.index,-1)
smote_values = np.unique(smote_values).tolist()
del smote_values[0]
print(smote_values) #不足的的索引


# In[130]:


# 转为参数里的形式
strategy = {}
for i in smote_values:
    item = {i:1000}
    strategy.update(item)
print(strategy)


# In[134]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy=strategy,k_neighbors=1)
X_train, y_train = smote.fit_resample(X_train, y_train)
pd.Series(y_train).value_counts()


# In[135]:


X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)


# ## Machine learning model training

# ### Training four base learners: decision tree, random forest, extra trees, XGBoost

# #### Apply XGBoost

# In[136]:


xg = xgb.XGBClassifier(n_estimators = 10)
start_time = time.time()
xg.fit(X_train,y_train)
end_time = time.time()
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[137]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (Original)')


# In[138]:


output_df


# #### Hyperparameter optimization (HPO) of XGBoost using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[139]:


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

start_time = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

end_time = time.time()

print("XGBoost: Hyperopt estimated optimum {}".format(best))


# In[140]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'learning_rate':  abs(float(best['learning_rate'])),
}
xg = xgb.XGBClassifier(**params)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[60]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (BO-TPE)')


# In[61]:


xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)


# #### Hyperparameter optimization (HPO) of XGBoost using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[62]:


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

start_time = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

end_time = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[ ]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'learning_rate': abs(float(optimal_configuration['learning_rate']))
}
xg = xgb.XGBClassifier(**params)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (PSO)')


# #### Hyperparameter optimization (HPO) of XGBoost using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


#Xgboost
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(10,100),
    'max_depth': range(4,100),
    'learning_rate': [i/100 for i in range(1, 90)]
}
              
ga = TPOTClassifier(generations= 3, population_size= 10, offspring_size= 5,
                                 verbosity= 3, early_stop= 5,
                                 config_dict=
                                 {'xgboost.XGBClassifier': parameters}, 
                                 cv = 3, scoring = 'accuracy')
start_time = time.time()
ga.fit(X_combined, y_combined)
end_time = time.time()


# In[ ]:


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


# In[ ]:


xg = xgb.XGBClassifier(**get_ga_optimized_parameters(ga, 'XGBClassifier'))
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('XGBoost (GA)')


# #### Apply RF

# In[ ]:


rf = RandomForestClassifier(random_state = 0)
start_time = time.time()
rf.fit(X_train,y_train)
end_time = time.time() 
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (Original)')


# #### Hyperparameter optimization (HPO) of random forest using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
    "max_features":hp.quniform('max_features', 1, 10, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion', available_criterion)
}

start_time = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

end_time = time.time()

print("Random Forest: Hyperopt estimated optimum {}".format(best))


# In[ ]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
rf_hpo = RandomForestClassifier(**params)
rf_hpo.fit(X_train,y_train)
rf_score=rf_hpo.score(X_test,y_test)
y_predict=rf_hpo.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (BO-TPE)')


# In[ ]:


rf_train=rf_hpo.predict(X_train)
rf_test=rf_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of random forest using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
    'max_features': [1, 10],
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

start_time = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

end_time = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[ ]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
rf_hpo = RandomForestClassifier(**params)
rf_hpo.fit(X_train,y_train)
rf_score=rf_hpo.score(X_test,y_test)
y_predict=rf_hpo.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (PSO)')


# #### Hyperparameter optimization (HPO) of random forest using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(20,200),
    "max_features":range(1,10),
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
start_time = time.time()
ga.fit(X_combined, y_combined)
end_time = time.time()


# In[ ]:


rf_hpo = RandomForestClassifier(**get_ga_optimized_parameters(ga, 'RandomForestClassifier'))
rf_hpo.fit(X_train,y_train)
rf_score=rf_hpo.score(X_test,y_test)
y_predict=rf_hpo.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': rf_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('RF (GA)')


# #### Apply DT

# In[ ]:


dt = DecisionTreeClassifier(random_state = 0)
start_time = time.time()
dt.fit(X_train,y_train)
end_time = time.time() 
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (Original)')


# #### Hyperparameter optimization (HPO) of decision tree using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
    "max_features":hp.quniform('max_features', 1, 10, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion',available_criterion)
}

start_time = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

end_time = time.time()

print("Decision tree: Hyperopt estimated optimum {}".format(best))


# In[ ]:


params = {
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
dt_hpo = DecisionTreeClassifier(**params)
dt_hpo.fit(X_train,y_train)
dt_score=dt_hpo.score(X_test,y_test)
y_predict=dt_hpo.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (BO-TPE)')


# In[ ]:


dt_train=dt_hpo.predict(X_train)
dt_test=dt_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of decision tree using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


#Random Forest
import optunity
import optunity.metrics

data=X_train
labels=y_train.tolist()
Y_train = y_train
Y_test = y_test
# Define the hyperparameter configuration space
search = {
    'max_features': [1, 10],
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

start_time = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

end_time = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[ ]:


params = {
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
dt_hpo = DecisionTreeClassifier(**params)
dt_hpo.fit(X_train,y_train)
dt_score=dt_hpo.score(X_test,y_test)
y_predict=dt_hpo.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (PSO)')


# #### Hyperparameter optimization (HPO) of decision tree using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    "max_features":range(1,10),
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
start_time = time.time()
ga.fit(X_combined, y_combined)
end_time = time.time()


# In[ ]:


dt_hpo = DecisionTreeClassifier(**get_ga_optimized_parameters(ga, 'DecisionTreeClassifier'))
dt_hpo.fit(X_train,y_train)
dt_score=dt_hpo.score(X_test,y_test)
y_predict=dt_hpo.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': dt_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('DT (GA)')


# #### Apply ET

# In[ ]:


et = ExtraTreesClassifier(random_state = 0)
start_time = time.time()
et.fit(X_train,y_train)
end_time = time.time() 
et_score=et.score(X_test,y_test)
y_predict=et.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (Original)')


# #### Hyperparameter optimization (HPO) of extra trees using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
    "max_features":hp.quniform('max_features', 1, 10, 1),
    "min_samples_split":hp.quniform('min_samples_split',2,11,1),
    "min_samples_leaf":hp.quniform('min_samples_leaf',1,11,1),
    "criterion":hp.choice('criterion',available_criterion)
}

start_time = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

end_time = time.time()

print("Random Forest: Hyperopt estimated optimum {}".format(best))


# In[ ]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    "min_samples_split":int(best['min_samples_split']),
    "min_samples_leaf":int(best['min_samples_leaf']),
    "criterion":available_criterion[int(best['criterion'])]
}
et_hpo = ExtraTreesClassifier(**params)
et_hpo.fit(X_train,y_train) 
et_score=et_hpo.score(X_test,y_test)
y_predict=et_hpo.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (BO-TPE)')


# In[ ]:


et_train=et_hpo.predict(X_train)
et_test=et_hpo.predict(X_test)


# #### Hyperparameter optimization (HPO) of extra trees using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
    'max_features': [1, 10],
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

start_time = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

end_time = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[ ]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'min_samples_leaf': int(optimal_configuration['min_samples_leaf']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'min_samples_split': int(optimal_configuration['min_samples_split']), 
    'max_features': int(optimal_configuration['max_features']), 
    'criterion': available_criterion[int(optimal_configuration['criterion']+0.5)]
}
et_hpo = ExtraTreesClassifier(**params)
et_hpo.fit(X_train,y_train) 
et_score=et_hpo.score(X_test,y_test)
y_predict=et_hpo.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (PSO)')


# #### Hyperparameter optimization (HPO) of extra trees using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


#Random Forest
from tpot import TPOTClassifier
# Define the hyperparameter configuration space
parameters = {
    'n_estimators': range(20,200),
    "max_features":range(1,10),
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
start_time = time.time()
ga.fit(X_combined, y_combined)
end_time = time.time()


# In[ ]:


et_hpo = ExtraTreesClassifier(**get_ga_optimized_parameters(ga, 'ExtraTreesClassifier'))
et_hpo.fit(X_train,y_train) 
et_score=et_hpo.score(X_test,y_test)
y_predict=et_hpo.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': et_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('ET (GA)')


# ### Apply stacking

# In[ ]:


base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
base_predictions_train.head(5)


# In[ ]:


dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)


# In[ ]:


dt_train.shape


# In[ ]:


x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)


# In[ ]:


start_time = time.time()
stk = xgb.XGBClassifier().fit(x_train, y_train)
end_time = time.time()
y_predict=stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': stk_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (Original)')


# #### Hyperparameter optimization (HPO) of the stacking ensemble model (XGBoost) using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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

start_time = time.time()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

end_time = time.time()

print("XGBoost: Hyperopt estimated optimum {}".format(best))


# In[ ]:


params = {
    'n_estimators': int(best['n_estimators']), 
    'max_depth': int(best['max_depth']),
    'learning_rate':  abs(float(best['learning_rate'])),
}
xg = xgb.XGBClassifier(**params)
xg.fit(x_train,y_train)
xg_score=xg.score(x_test,y_test)
y_predict=xg.predict(x_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (BO-TPE)')


# #### Hyperparameter optimization (HPO) of stacking ensemble model (XGBoost) using Particle Swarm Optimization (PSO)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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

start_time = time.time()

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )

end_time = time.time()

print(optimal_configuration)
print("Accuracy:"+ str(info.optimum))


# In[ ]:


params = {
    'n_estimators': int(optimal_configuration['n_estimators']), 
    'max_depth': int(optimal_configuration['max_depth']), 
    'learning_rate': abs(float(optimal_configuration['learning_rate']))
}
xg = xgb.XGBClassifier(**params)
xg.fit(x_train,y_train)
xg_score=xg.score(x_test,y_test)
y_predict=xg.predict(x_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (PSO)')


# #### Hyperparameter optimization (HPO) of stacking ensemble model (XGBoost) using Genetic Algorithm (GA)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

# In[ ]:


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
start_time = time.time()
ga.fit(X_combined, y_combined)
end_time = time.time()


# In[ ]:


xg = xgb.XGBClassifier(**get_ga_optimized_parameters(ga, 'XGBClassifier'))
xg.fit(x_train,y_train)
xg_score=xg.score(x_test,y_test)
y_predict=xg.predict(x_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))


# In[ ]:


# Add to output sheet
result_dict = {
    'Accuracy': xg_score,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Time': end_time-start_time
}
output_df = output_df.append(result_dict, ignore_index=True)
# Add index name
output_index.append('Stacking (GA)')


# In[ ]:


# Rename the index
output_df.index = output_index
# Save the result to file
import datetime
output_df.to_excel('result-{}.xlsx'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))


# In[59]:


import requests
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                     json={
                         "token": "5ac74bd3b65f",
                         "title": "MTH KDDCUP99",
                         "content": "+1遍"
                     })
print(resp.content.decode())

