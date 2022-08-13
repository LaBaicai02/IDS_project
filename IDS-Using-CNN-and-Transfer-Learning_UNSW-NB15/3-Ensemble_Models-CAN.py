#!/usr/bin/env python
# coding: utf-8

# # A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles 
# This is the code for the paper entitled "**A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles**" accepted in IEEE International Conference on Communications (IEEE ICC).  
# Authors: Li Yang (lyang339@uwo.ca) and Abdallah Shami (Abdallah.Shami@uwo.ca)  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# **Notebook 3: Ensemble Models**  
# Aims:  construct three ensemble techniques: Bagging, Probability averaging, and Concatenation, to further improve prediction accuracy  
# * Bagging: use majority voting of top single models  
# * Probability averaging: calculate the average probability of the single model prediction results (the last layer of CNN models), and select the largest probability class to be the final class  
# * Concatenation: extract the features in the last several layers of single models, and concatenate together to generate the new layers, and add a dense layer to do prediction

# ## Import libraries

# In[1]:


from collections import defaultdict
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import concatenate,Dense,Flatten,Dropout
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import plot_model
import datetime
import math
import numpy as np
import operator
import os
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.callbacks as kcallbacks
import time
import warnings
warnings.filterwarnings("ignore")


# ## Read the test set

# In[2]:


#generate images from train set and validation set
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=128

test_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = test_datagen.flow_from_directory(
        './test_224/',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')

num_class = validation_generator.num_classes


# In[3]:


#generate labels indicating disease (1) or normal (0)
label=validation_generator.class_indices
label={v: k for k, v in label.items()}


# In[4]:


print(label)


# In[5]:


#read images from validation folder
rootdir = './test_224/'
test_labels = []
test_images=[]
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
            continue
        test_labels.append(subdir.split('/')[-1])
        test_images.append(os.path.join(subdir, file))
        
print(test_labels[0],test_images[0])


# ## Load 6 trained CNN models

# In[7]:


#load model 1: xception
xception_model=load_model('./xception.h5')


# In[9]:


#load model 2: VGG16
vgg_model=load_model('./VGG16.h5')


# In[10]:


#load model 3: VGG19
vgg19_model=load_model('./VGG19.h5')


# In[11]:


#load model 4: inception
incep_model=load_model('./inception.h5')


# In[12]:


#load model 5: inceptionresnet
inres_model=load_model('./inceptionresnet.h5')


# In[ ]:


#load model 6: resnet
res_model=load_model('./resnet.h5')


# ## Use the original CNN base models to make predictions

# In[ ]:


class output_sheet:
    def __init__(self, columns:list=['accuracy', 'precision', 'recall', 'f1-score', 'training_time', 'testing_time']):
        self.output_df = pd.DataFrame(columns=columns)
        # self.output_index = list()
    def add(self, item:str, **values):
        # self.output_df = self.output_df.append(values, ignore_index=True)
        temp = pd.DataFrame(values, columns=self.output_df.columns.to_list(), index=[item])
        self.output_df = pd.concat([self.output_df, temp], axis=0)
        # self.output_index.append(item)
    # def apply_index(self):
    #     self.output_df.index = self.output_index
    def to_excel(self, path=None):
        if path is None: path='3-result-{}.xlsx'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
        # self.apply_index()
        self.output_df.to_excel(path)


# In[ ]:


output_obj = output_sheet(columns=['accuracy', 'precision', 'recall', 'f1-score', 'training_time', 'testing_time'])


# ### 1. Xception

# In[20]:


#Single image prediction
import cv2
import matplotlib.pyplot as plt
test=cv2.imread(test_images[0])

img_show=test[:,:,[2,1,0]]
test=test/255.
test_shape=(1,)+test.shape
test=test.reshape(test_shape)

res=xception_model.predict(test)

prob=res[0,np.argmax(res,axis=1)[0]]
res=label[np.argmax(res,axis=1)[0]]
print('Predicted result for the first image: %s'%res)
print('Confidence level: %s'%prob)
plt.imshow(img_show)
plt.show()


# In[12]:


# %%time
import time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    xception_model_batch=xception_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=xception_model_batch[0,np.argmax(xception_model_batch,axis=1)[0]]
    res=label[np.argmax(xception_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()    


# In[13]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Xception accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('Xception', **values)


# ### 2. VGG16

# In[14]:


# %%time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    vgg_model_batch=vgg_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=vgg_model_batch[0,np.argmax(vgg_model_batch,axis=1)[0]]
    res=label[np.argmax(vgg_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()     


# In[15]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('VGG16 accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('VGG16', **values)


# ### 3. VGG19

# In[16]:


# %%time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    vgg19_model_batch=vgg19_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=vgg19_model_batch[0,np.argmax(vgg19_model_batch,axis=1)[0]]
    res=label[np.argmax(vgg19_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()


# In[17]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('VGG19 accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('VGG19', **values)


# ### 4. Inception

# In[18]:


# %%time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    incep_model_batch=incep_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=incep_model_batch[0,np.argmax(incep_model_batch,axis=1)[0]]
    res=label[np.argmax(incep_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()


# In[19]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('inception accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('Inception', **values)


# ### 5. InceptionResnet

# In[20]:


# %%time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    inres_model_batch=inres_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=inres_model_batch[0,np.argmax(inres_model_batch,axis=1)[0]]
    res=label[np.argmax(inres_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()


# In[21]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('inceptionresnet accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('InceptionResnet', **values)


# ### 6. Resnet

# In[23]:


# %%time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(length):
    inputimg=test_images[i]
    test_batch=[]
    thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
    #print(thisimg)
    test_shape=(1,)+thisimg.shape
    thisimg=thisimg.reshape(test_shape)
    res_model_batch=res_model.predict(thisimg) #use master model to process the input image
    #generate result by model 1
    prob=res_model_batch[0,np.argmax(res_model_batch,axis=1)[0]]
    res=label[np.argmax(res_model_batch,axis=1)[0]]
    predict.append(res)
t2 = time.time()


# In[24]:


# %%time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('resnet accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('Resnet', **values)


# Best performing single model (vgg):  
# Accuracy: 99.96

# # Bagging ensemble

# In[25]:


import time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(((length-1)//127)+1):
    inputimg=test_images[127*i:127*(i+1)]
    test_batch=[]
    for path in inputimg:
        thisimg=np.array(Image.open(path))/255
        test_batch.append(thisimg)
    #generate result by model 1
    xception_model_batch=xception_model.predict(np.array(test_batch))
    xception_model_batch=list(np.argmax(xception_model_batch,axis=1))
    xception_model_batch=[label[con] for con in xception_model_batch]
#     print(xception_model_batch)
    #generate result by model 2
    vgg_model_batch=vgg_model.predict(np.array(test_batch))
    vgg_model_batch=list(np.argmax(vgg_model_batch,axis=1))
    vgg_model_batch=[label[con] for con in vgg_model_batch]
#     print(vgg_model_batch)
    #generate result by model 3
    vgg19_model_batch=vgg19_model.predict(np.array(test_batch))
    vgg19_model_batch=list(np.argmax(vgg19_model_batch,axis=1))
    vgg19_model_batch=[label[con] for con in vgg19_model_batch]
#     print(vgg19_model_batch)
    #generate result by model 4
    incep_model_batch=incep_model.predict(np.array(test_batch))
    incep_model_batch=list(np.argmax(incep_model_batch,axis=1))
    incep_model_batch=[label[con] for con in incep_model_batch]
#     print(incep_model_batch)
    #generate result by model 5
    inres_model_batch=inres_model.predict(np.array(test_batch))
    inres_model_batch=list(np.argmax(inres_model_batch,axis=1))
    inres_model_batch=[label[con] for con in inres_model_batch]
#     print(inres_model_batch)
    #bagging the three results generated by 3 singular models
    predict_batch=[]
    for i,j,k,p,q in zip(xception_model_batch,vgg_model_batch,vgg19_model_batch,incep_model_batch,inres_model_batch):
        count=defaultdict(int)
        count[i]+=1
        count[j]+=1
        count[k]+=1
        count[p]+=1
        count[q]+=1
        #rank the predicted results in descending order
        predict_one=sorted(count.items(), key=operator.itemgetter(1),reverse=True)[0][0]
        predict_batch.append(predict_one)
#     print('predict:',predict_batch)
    predict.append(predict_batch)
t2 = time.time()
print('The testing time is :%f seconds' % (t2-t1))


# In[26]:


predict=sum(predict,[])


# In[27]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('bagging accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'testing_time': t2-t1
}
output_obj.add('Bagging Ensemble', **values)


# After bagging ensemble, the accuracy improved to 0.990

# # Probability Averaging

# In[29]:


from collections import defaultdict
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import concatenate,Dense,Flatten,Dropout,Average
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import plot_model
import math
import numpy as np
import operator
import os
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as kcallbacks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[30]:


import time
t1 = time.time()
img=Input(shape=(224,224,3),name='img')
feature1=xception_model(img)
feature2=vgg_model(img)
feature3=incep_model(img)
for layer in xception_model.layers:  
    layer.trainable = False 
for layer in vgg_model.layers:  
    layer.trainable = False  
for layer in incep_model.layers:  
    layer.trainable = False  
output=Average()([feature1,feature2,feature3]) #add the confidence lists generated by 3 models
model=Model(inputs=img,outputs=output)

#the optimization function
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
t2 = time.time()
print('The training time is :%f seconds' % (t2-t1))


# In[31]:


#read images from validation folder
rootdir = './test_224/'
test_labels = []
test_images=[]
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
            continue
        test_labels.append(subdir.split('/')[-1])
        test_images.append(os.path.join(subdir, file))
        
print(test_labels[0],test_images[0])


# In[32]:


#test the averaging model on the validation set
import time
predict=[]
length=len(test_images)
t3 = time.time()
for i in range(((length-1)//127)+1):
    inputimg=test_images[127*i:127*(i+1)]
    test_batch=[]
    for path in inputimg:
        thisimg=np.array(Image.open(path))/255
        test_batch.append(thisimg)
    #print(i, np.array(test_batch).shape)
    model_batch=model.predict(np.array(test_batch))
    predict_batch=list(np.argmax(model_batch,axis=1))
    predict_batch=[label[con] for con in predict_batch]
    predict.append(predict_batch)

predict=sum(predict,[])

t4 = time.time()
print('The testing time is :%f seconds' % (t4-t3))


# In[33]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Probability Averaging accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)


# In[34]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'training_time': t2-t1,
    'testing_time': t4-t3
}
output_obj.add('Probability Averaging', **values)


# # Concatenation

# In[35]:


from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import concatenate,Dense,Flatten,Dropout
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import plot_model
import math
import os
import tensorflow.keras as keras
import tensorflow.keras.callbacks as kcallbacks


# In[36]:


# for i,layer in enumerate(xception_model.layers):
#     print(i,layer.name)


# In[37]:


# for i,layer in enumerate(vgg_model.layers):
#     print(i,layer.name)


# In[38]:


# for i,layer in enumerate(vgg19_model.layers):
#     print(i,layer.name)


# In[39]:


# for i,layer in enumerate(incep_model.layers):
#     print(i,layer.name)


# In[40]:


# for i,layer in enumerate(inres_model.layers):
#     print(i,layer.name)


# ### Construct the ensemble model using the last "dense layer" of each base CNN model

# In[ ]:


def get_last_layer(model, prefix='dense'):
    target_layer = None
    for layer in model.layers:
        if layer.name.startswith(prefix): target_layer=layer
    return target_layer


# In[43]:



model1=Model(inputs=[xception_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=xception_model, prefix='dense').output,name='xception')
model2=Model(inputs=[vgg_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=vgg_model, prefix='dense').output,name='vgg')
model3=Model(inputs=[vgg19_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=vgg19_model, prefix='dense').output,name='vgg19')
model4=Model(inputs=[incep_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=incep_model, prefix='dense').output,name='incep')
model5=Model(inputs=[inres_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=inres_model, prefix='dense').output,name='inres')


# In[44]:


#plot the figures
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
    def get_best(self, target_type:str='epoch'):
        # Get the index of the best record
        max_index = self.accuracy[target_type].index(max(self.accuracy[target_type]))
        # Return the accuracy, loss, val_acc, val_loss of the best record
        return {
            'accuracy': self.accuracy[target_type][max_index], 
            'loss': self.losses[target_type][max_index], 
            'val_acc': self.val_acc[target_type][max_index], 
            'val_loss': self.val_loss[target_type][max_index]
            }


# In[45]:


ensemble_history= LossHistory()


# In[ ]:


# Measure the running time of model training
class TimeMeasurement(keras.callbacks.Callback):
    def __init__(self):
        self.start_time=None
        self.stop_time=None
    # Start timing when trainning starts
    def on_train_begin(self, logs=None):
        self.start_time=time.time()
        self.stop_time=None
    # Stop timing when trainning ends
    def on_train_end(self, logs=None):
        self.stop_time=time.time()
        if self.start_time is None: print("Time Measuring Failed")
    # Get processing time
    def get_processing_time(self):
        if (self.start_time is None or self.stop_time is None): raise Exception("Wrong Time Measurement")
        else: return self.stop_time-self.start_time


# In[ ]:


timer = TimeMeasurement()


# In[46]:


#generate training and test images
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=128	#could try 128 or 32

#Normalization
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './train_224/',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        './test_224/',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')
num_class = validation_generator.num_classes


# In[47]:


def lr_decay(epoch):
    lrs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001,0.00001,0.000001,
           0.000001,0.000001,0.000001,0.000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001
          ]
    return lrs[epoch]


# In[48]:


auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
my_lr = LearningRateScheduler(lr_decay)


# In[49]:


def ensemble(num_class,epochs,savepath='./ensemble.h5'):
    img=Input(shape=(224,224,3),name='img')
    feature1=model1(img)
    feature2=model2(img)
    feature3=model3(img)
    x=concatenate([feature1,feature2,feature3])
    x=Dropout(0.5)(x)
    x=Dense(64,activation='relu')(x)
    x=Dropout(0.25)(x)
    output=Dense(num_class,activation='softmax',name='output')(x)
    model=Model(inputs=img,outputs=output)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #train model
    earlyStopping=kcallbacks.EarlyStopping(monitor='val_acc',patience=2, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    hist=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[earlyStopping,saveBestModel,ensemble_history,auto_lr,timer],
    )


# In[50]:


ensemble_model=ensemble(num_class=num_class,epochs=20)


# In[51]:


ensemble_model=load_model('./ensemble.h5')


# In[52]:


#read images from validation folder
rootdir = './test_224/'
test_labels = []
test_images=[]
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
            continue
        test_labels.append(subdir.split('/')[-1])
        test_images.append(os.path.join(subdir, file))
        
print(test_labels[0],test_images[0])


# In[53]:


#test the averaging model on the validation set
import time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(((length-1)//127)+1):
    inputimg=test_images[127*i:127*(i+1)]
    test_batch=[]
    for path in inputimg:
        thisimg=np.array(Image.open(path))/255
        test_batch.append(thisimg)
    #print(i, np.array(test_batch).shape)
    ensemble_model_batch=ensemble_model.predict(np.array(test_batch))
    predict_batch=list(np.argmax(ensemble_model_batch,axis=1))
    predict_batch=[label[con] for con in predict_batch]
    predict.append(predict_batch)

predict=sum(predict,[])

t2 = time.time()
print('The testing time is :%f seconds' % (t2-t1))


# In[54]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Concatenation accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)


# In[55]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_labels, predict))
target_names = label.values()
print(classification_report(test_labels, predict, target_names=target_names))


# In[ ]:


values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'training_time': timer.get_processing_time(),
    'testing_time': t2-t1
}
output_obj.add('Concatenation', **values)


# In[ ]:


output_obj.to_excel()

