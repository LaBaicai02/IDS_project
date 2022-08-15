# %% [markdown]
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

# %% [markdown]
# ## Import libraries

# %%
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Read the test set

# %%
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

# %%
#generate labels indicating disease (1) or normal (0)
label=validation_generator.class_indices
label={v: k for k, v in label.items()}

# %%
print(label)

# %%
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

# %%
len_test = len(test_labels)

# %% [markdown]
# ## Load 6 trained CNN models

# %%
 #load model 1: xception
xception_model=load_model('./xception.h5')

# %%
 #load model 2: VGG16
vgg_model=load_model('./VGG16.h5')

# %%
 #load model 3: VGG19
vgg19_model=load_model('./VGG19.h5')

# %%
 #load model 4: inception
incep_model=load_model('./inception.h5')

# %%
 #load model 5: inceptionresnet
inres_model=load_model('./inceptionresnet.h5')

# %%
 #load model 6: resnet
res_model=load_model('./resnet.h5')

# %% [markdown]
# ## Use the original CNN base models to make predictions

# %%
# Prepare the output dir
output_dir = 'output/CNN_based/3-output-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
img_dir = os.path.join(output_dir, 'img')
os.makedirs(img_dir)
# Prepare the log file
log_file = open(os.path.join(output_dir, 'classification_report-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))), 'w+')

# %%
class output_sheet:
    def __init__(self, columns:list=['accuracy', 'precision', 'recall', 'f1-score', 'training_time', 'predict_time_per_image']):
        self.output_df = pd.DataFrame(columns=columns)
        # self.output_index = list()
    def add(self, item:str, **values:dict):
        # self.output_df = self.output_df.append(values, ignore_index=True)
        temp = pd.DataFrame(values, columns=self.output_df.columns.to_list(), index=[item])
        self.output_df = pd.concat([self.output_df, temp], axis=0)
        # self.output_index.append(item)
    # def apply_index(self):
    #     self.output_df.index = self.output_index
    def to_excel(self, path=None):
        if path is None: path=os.path.join(output_dir, '2-result-{}.xlsx'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
        # self.apply_index()
        self.output_df.to_excel(path)

# %%
output_obj = output_sheet(columns=['accuracy', 'precision', 'recall', 'f1-score', 'training_time', 'predict_time_per_image'])

# %%
# Prediction function
def get_prediction(model, test_images=test_images, label=label, batch_size=BATCHSIZE, verbose='auto'):
    predict=[]
    length=len(test_images)
    for i in range(((length-1)//batch_size)+1):
        inputimg=test_images[batch_size*i:batch_size*(i+1)]
        test_batch=[]
        for path in inputimg:
            thisimg=np.array(Image.open(path))/255
            test_batch.append(thisimg)
        model_batch=model.predict(np.array(test_batch), verbose=verbose) #use master model to process the input image
        predict_batch=list(np.argmax(model_batch,axis=1))
        predict_batch=[label[con] for con in predict_batch]
        predict.extend(predict_batch)
    return predict

# %%
def generate_report(name:str, y_pred, y_true=test_labels, label=label, img_dir=img_dir, log_file=log_file, figsize=(18,14), log_classification_report:bool=True, save_img:bool=True, print_classifiaction_report:bool=True, display_confusion_matrix:bool=False):
    # Generate classification report
    report_str = classification_report(y_true,y_pred,zero_division=0)
    if log_classification_report and log_file.writable(): log_file.write('******{}******\n'.format(name)+report_str+'\n')
    if print_classifiaction_report: print(report_str)
    # Generate confusion matrix
    cm=confusion_matrix(y_true,y_pred)
    f,ax=plt.subplots(figsize=figsize)
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    ax.set_xticklabels(label.values())
    ax.set_yticklabels(label.values())
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    if save_img: plt.savefig(os.path.join(img_dir, '{}.pdf'.format(name)))
    if display_confusion_matrix: plt.show()

# %% [markdown]
# ### 1. Xception

# %%
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

# %%
# %%time
import time
t1 = time.time()
predict = get_prediction(xception_model)
t2 = time.time()    

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Xception accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Xception', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('Xception', **values)

# %% [markdown]
# ### 2. VGG16

# %%
# %%time
t1 = time.time()
prediction = get_prediction(vgg_model)
t2 = time.time()     

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('VGG16 accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('VGG16', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('VGG16', **values)

# %% [markdown]
# ### 3. VGG19

# %%
# %%time
t1 = time.time()
prediction = get_prediction(vgg19_model)
t2 = time.time()

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('VGG19 accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('VGG19', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('VGG19', **values)

# %% [markdown]
# ### 4. Inception

# %%
# %%time
t1 = time.time()
prediction = get_prediction(incep_model)
t2 = time.time()

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('inception accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Inception', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('Inception', **values)

# %% [markdown]
# ### 5. InceptionResnet

# %%
# %%time
t1 = time.time()
prediction = get_prediction(inres_model)
t2 = time.time()

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('inceptionresnet accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('InceptionResnet', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('InceptionResnet', **values)

# %% [markdown]
# ### 6. Resnet

# %%
# %%time
t1 = time.time()
prediction = get_prediction(res_model)
t2 = time.time()

# %%
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
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Resnet', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('Resnet', **values)

# %% [markdown]
# Best performing single model (vgg):  
# Accuracy: 99.96

# %% [markdown]
# # Bagging ensemble

# %%
import time
predict=[]
length=len(test_images)
t1 = time.time()
for i in range(((length-1)//BATCHSIZE)+1):
    inputimg=test_images[BATCHSIZE*i:BATCHSIZE*(i+1)]
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
    predict.extend(predict_batch)
t2 = time.time()
print('The testing time is :%f seconds' % (t2-t1))

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('bagging accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)

# %%
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Bagging_ensemble', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('Bagging Ensemble', **values)

# %% [markdown]
# After bagging ensemble, the accuracy improved to 0.990

# %% [markdown]
# # Probability Averaging

# %%
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

# %%
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

# %%
#test the averaging model on the validation set
import time
t3 = time.time()
predict = get_prediction(model)
t4 = time.time()
print('The testing time is :%f seconds' % (t4-t3))

# %%
from sklearn.metrics import accuracy_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Probability Averaging accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)

# %%
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Probability_averaging', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'training_time': t2-t1,
    'predict_time_per_image': (t4-t3)/len_test
}
output_obj.add('Probability Averaging', **values)

# %% [markdown]
# # Concatenation

# %%
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

# %%
# for i,layer in enumerate(xception_model.layers):
#     print(i,layer.name)

# %%
# for i,layer in enumerate(vgg_model.layers):
#     print(i,layer.name)

# %%
# for i,layer in enumerate(vgg19_model.layers):
#     print(i,layer.name)

# %%
# for i,layer in enumerate(incep_model.layers):
#     print(i,layer.name)

# %%
# for i,layer in enumerate(inres_model.layers):
#     print(i,layer.name)

# %% [markdown]
# ### Construct the ensemble model using the last "dense layer" of each base CNN model

# %%
def get_last_layer(model, prefix='dense'):
    target_layer = None
    for layer in model.layers:
        if layer.name.startswith(prefix): target_layer=layer
    return target_layer

# %%

model1=Model(inputs=[xception_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=xception_model, prefix='dense').output,name='xception')
model2=Model(inputs=[vgg_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=vgg_model, prefix='dense').output,name='vgg')
model3=Model(inputs=[vgg19_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=vgg19_model, prefix='dense').output,name='vgg19')
model4=Model(inputs=[incep_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=incep_model, prefix='dense').output,name='incep')
model5=Model(inputs=[inres_model.layers[0].get_input_at(0)],outputs=get_last_layer(model=inres_model, prefix='dense').output,name='inres')

# %%
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

# %%
ensemble_history= LossHistory()

# %%
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

# %%
timer = TimeMeasurement()

# %%
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './train_224/',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')

# %%
def lr_decay(epoch):
    lrs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001,0.00001,0.000001,
           0.000001,0.000001,0.000001,0.000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001,0.0000001
          ]
    return lrs[epoch]

# %%
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
my_lr = LearningRateScheduler(lr_decay)

# %%
def ensemble(num_class,epochs,savepath='./ensemble.h5'):
    img=Input(shape=INPUT_SIZE,name='img')
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
    earlyStopping=kcallbacks.EarlyStopping(monitor='val_acc',patience=2, verbose=1, mode='auto', restore_best_weights=True)
    saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    hist=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[earlyStopping,saveBestModel,ensemble_history,auto_lr,timer],
    )
    return model

# %%
ensemble_model=ensemble(num_class=num_class,epochs=20)

# %%
# ensemble_model=load_model('./ensemble.h5')

# %%
#test the averaging model on the validation set
import time
t1 = time.time()
predict = get_prediction(ensemble_model)
t2 = time.time()
print('The testing time is :%f seconds' % (t2-t1))

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
acc=accuracy_score(test_labels,predict)
pre=precision_score(test_labels,predict,average='weighted')
re=recall_score(test_labels,predict,average='weighted')
f1=f1_score(test_labels,predict,average='weighted')
print('Concatenation accuracy: %s'%acc)
print('precision: %s'%pre)
print('recall: %s'%re)
print('f1: %s'%f1)

# %%
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(test_labels, predict))
# target_names = label.values()
# print(classification_report(test_labels, predict, target_names=target_names))
generate_report('Concatenation', predict)

# %%
values = {
    'accuracy': acc,
    'precision': pre,
    'recall': re,
    'f1-score': f1,
    'training_time': timer.get_processing_time(),
    'predict_time_per_image': (t2-t1)/len_test
}
output_obj.add('Concatenation', **values)

# %%
output_obj.to_excel()
log_file.close()


