# %% [markdown]
# # A Transfer Learning and Optimized CNN Based Intrusion Detection System using UNSW-NB15
# This is the code for the paper entitled "**A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles**" accepted in IEEE International Conference on Communications (IEEE ICC).  
# Authors: Li Yang (lyang339@uwo.ca) and Abdallah Shami (Abdallah.Shami@uwo.ca)  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# **Notebook 1: Data pre-processing**  
# Procedures:  
# &nbsp; 1): Read the dataset  
# &nbsp; 2): Transform the tabular data into images  
# &nbsp; 3): Display the transformed images  
# &nbsp; 4): Split the training and test set  

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Read the Car-Hacking/CAN-Intrusion dataset
# The complete Car-Hacking dataset is publicly available at: https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset  
# In this repository, due to the file size limit of GitHub, we use the 5% subset.

# %%
#Read dataset
df=pd.read_csv('data/UNSW-NB15.csv')

# %%
df

# %%
df['attack_cat'].value_counts()

# %%
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

# %%
# Only keep the influencing features
df = df[features]

# %% [markdown]
# ### Data Sampling

# %%
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col_name in df.dtypes[df.dtypes == 'object'].index.drop(['attack_cat']):
      df[col_name] = labelencoder.fit_transform(df[col_name])
df.insert(loc=len(df.columns), column='tmp_label', value=labelencoder.fit_transform(df['attack_cat']))

# %%
df['tmp_label'].value_counts()

# %%
# retain the minority class instances and sample the majority class instances
split_threshold = 10000
label_col = 'tmp_label'
df_minor = pd.DataFrame(columns=df.columns.to_list())
for cat_num, record_num in df[label_col].value_counts().to_dict().items():
    if record_num < split_threshold: df_minor=df_minor.append(df[df[label_col]==cat_num])
# df_minor = df[(df['tmp_label']==0)|(df['tmp_label']==1)|(df['tmp_label']==9)|(df['tmp_label']==2)|(df['tmp_label']==10)]
df_major = df.drop(df_minor.index)

# %%
X = df_major.drop(['attack_cat', 'tmp_label'],axis=1) 
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)

# %%
# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)

# %%
klabel=kmeans.labels_
df_major['klabel']=klabel

# %%
df_major['klabel'].value_counts()

# %%
len(df.columns.to_list())

# %%
cols = list(df_major)
cols.insert(len(cols)-2, cols.pop(cols.index('tmp_label')))
df_major = df_major.loc[:, cols]

# %%
df_major

# %%
def typicalSampling(group):
    name = group.name
    frac = 0.25
    return group.sample(frac=frac)

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)

# %%
result['attack_cat'].value_counts()

# %%
result

# %%
result = result.drop(['tmp_label', 'klabel'],axis=1)
df = result.append(df_minor.drop(['tmp_label'], axis=1))

# %%
df.to_csv('./data/UNSW-NB15_sample_km.csv',index=0)

# %%
# The labels of the dataset. "R" indicates normal patterns, and there are four types of attack (DoS, fuzzy. gear spoofing, and RPM spoofing zttacks)
df['attack_cat'].value_counts()

# %% [markdown]
# ### Feature Engineering

# %%
# df=pd.read_csv('./data/UNSW-NB15_sample_km.csv')

# %%
X = df.drop(['attack_cat'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

# %%
from sklearn.feature_selection import mutual_info_classif
importances = mutual_info_classif(X_train, y_train)

# %%
# calculate the sum of importance scores
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])

# %%
# select the important features from top to bottom until the accumulated importance reaches 90%
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2>=0.9:
        break        

# %%
fs.append('attack_cat')
df = df[fs]
df

# %% [markdown]
# ### SMOTE to solve class-imbalance

# %%
df['attack_cat'].value_counts()

# %%
sampling_strategy = dict()
smote_threshold = 4000
for key, value in df['attack_cat'].value_counts().to_dict().items():
    if value < smote_threshold: sampling_strategy[key]=smote_threshold

# %%
from imblearn.over_sampling import SMOTE
# smote=SMOTE(n_jobs=-1,sampling_strategy={'Backdoor':4000, 'Worms':4000, 'Shellcode': 4000, 'Analysis': 4000, 'Reconnaissance': 4000})
smote=SMOTE(n_jobs=-1,sampling_strategy=sampling_strategy)

# %%
x, y = smote.fit_resample(df.drop(['attack_cat'], axis=1), df['attack_cat'])

# %%
x.insert(loc=len(x.columns), column='attack_cat', value=y.values)
df = x

# %% [markdown]
# ## Data Transformation
# Convert tabular data to images
# Procedures:
# 1. Use quantile transform to transform the original data samples into the scale of [0,255], representing pixel values
# 2. Generate images for each category (Normal, DoS, Fuzzy, Gear, RPM), each image consists of 27 data samples with 9 features. Thus, the size of each image is 9*9*3, length 9, width 9, and 3 color channels (RGB).

# %%
# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer() 
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# %%
# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x*255))

# %%
df.describe()

# %% [markdown]
# All features are in the same scale of [0,255]

# %% [markdown]
# ### Generate images for each class

# %%
label_col = 'attack_cat'
labels = df[label_col].value_counts().index.to_list()
dfs = [df[df[label_col]==tmp].drop([label_col],axis=1) for tmp in labels]
# df0=df[df[label_col]=='Normal'].drop([label_col],axis=1)
# df1=df[df[label_col]=='Generic'].drop([label_col],axis=1)
# df2=df[df[label_col]=='Exploits'].drop([label_col],axis=1)
# df3=df[df[label_col]=='Fuzzers'].drop([label_col],axis=1)
# df4=df[df[label_col]=='DoS'].drop([label_col],axis=1)
# df5=df[df[label_col]=='Reconnaissance'].drop([label_col],axis=1)
# df6=df[df[label_col]=='Analysis'].drop([label_col],axis=1)
# df7=df[df[label_col]=='Backdoor'].drop([label_col],axis=1)
# df8=df[df[label_col]=='Shellcode'].drop([label_col],axis=1)
# df9=df[df[label_col]=='Backdoors'].drop([label_col],axis=1)
# df10=df[df[label_col]=='Worms'].drop([label_col],axis=1)

# %%
# Generate 9*9 color images for every class
# Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)
# In this case the n = 40
n = len(df.columns)-1
for i in range(len(dfs)):
    tmp_df = dfs[i]
    count=0
    ims = []

    image_path = "train/{}/".format(labels[i])
    os.makedirs(image_path)

    for i in range(0, len(tmp_df)):  
        count += 1
        if count<=n*3: 
            im=tmp_df.iloc[i].values
            ims=np.append(ims,im)
        else:
            ims=np.array(ims).reshape(n,n,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=1 # Fix bug
            ims = tmp_df.iloc[i].values # Fix bug

# %% [markdown]
# ### Display samples for each category

# %%
# Read the images for each category, the file name may vary (27.png, 83.png...)
# imgs = [Image.open('./train/{}/93.png'.format(i)) for i in range(len(dfs))]
imgs = list()
titles = list()
Train_Dir='./train/'
for dir in os.listdir(Train_Dir):
    img_files = os.listdir(os.path.join(Train_Dir, dir))
    if len(img_files) != 0: 
        imgs.append(Image.open(os.path.join(Train_Dir, dir, img_files[0])))
        titles.append(dir)
# img1 = Image.open('./train/0/123.png')
# img2 = Image.open('./train/1/123.png')
# img3 = Image.open('./train/2/123.png')
# img4 = Image.open('./train/3/123.png')
# img5 = Image.open('./train/4/123.png')
# img6 = Image.open('./train/0/123.png')
# img7 = Image.open('./train/1/123.png')
# img8 = Image.open('./train/2/123.png')
# img9 = Image.open('./train/3/123.png')
# img10 = Image.open('./train/4/123.png')
# img11 = Image.open('./train/4/123.png')
# titles = df[label_col].value_counts().index.to_list()

plot_row_num = (len(imgs)-1)//5+1
plt.figure(figsize=(15, plot_row_num*3)) 
for i in range(len(imgs)):
    plt.subplot(plot_row_num, 5, i+1)
    plt.imshow(imgs[i])
    plt.title(titles[i])

# plt.subplot(1,5,1)
# plt.imshow(img1)
# plt.title("Normal")
# plt.subplot(1,5,2)
# plt.imshow(img2)
# plt.title("RPM Spoofing")
# plt.subplot(1,5,3)
# plt.imshow(img3)
# plt.title("Gear Spoofing")
# plt.subplot(1,5,4)
# plt.imshow(img4)
# plt.title("DoS Attack")
# plt.subplot(1,5,5)
# plt.imshow(img5)
# plt.title("Fuzzy Attack")

plt.show()  # display it

# %% [markdown]
# ## Split the training and test set 

# %%
# Create folders to store images
Train_Dir='./train/'
Val_Dir='./test/'
allimgs=[]
for subdir in os.listdir(Train_Dir):
    for filename in os.listdir(os.path.join(Train_Dir,subdir)):
        filepath=os.path.join(Train_Dir,subdir,filename)
        allimgs.append(filepath)
print(len(allimgs)) # Print the total number of images

# %%
#split a test set from the dataset, train/test size = 80%/20%
Numbers=len(allimgs)//5 	#size of test set (20%)

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.move(srcfile,dstfile)          
        #print ("move %s -> %s"%(srcfile,dstfile))

# %%
# The size of test set
Numbers

# %%
# Create the test set
val_imgs=random.sample(allimgs,Numbers)
for img in val_imgs:
    dest_path=img.replace(Train_Dir,Val_Dir)
    mymovefile(img,dest_path)
print('Finish creating test set')

# %%
#resize the images 224*224 for better CNN training
def get_224(folder,dstdir):
    imgfilepaths=[]
    for root,dirs,imgs in os.walk(folder):
        for thisimg in imgs:
            thisimg_path=os.path.join(root,thisimg)
            imgfilepaths.append(thisimg_path)
    for thisimg_path in imgfilepaths:
        dir_name,filename=os.path.split(thisimg_path)
        dir_name=dir_name.replace(folder,dstdir)
        new_file_path=os.path.join(dir_name,filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img=cv2.imread(thisimg_path)
        img=cv2.resize(img,(224,224))
        cv2.imwrite(new_file_path,img)
    print('Finish resizing'.format(folder=folder))

# %%
DATA_DIR_224='./train_224/'
get_224(folder='./train/',dstdir=DATA_DIR_224)

# %%
DATA_DIR2_224='./test_224/'
get_224(folder='./test/',dstdir=DATA_DIR2_224)


