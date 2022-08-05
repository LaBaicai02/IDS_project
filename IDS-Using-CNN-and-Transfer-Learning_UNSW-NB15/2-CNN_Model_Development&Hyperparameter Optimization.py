#!/usr/bin/env python
# coding: utf-8

# # A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles 
# This is the code for the paper entitled "**A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles**" accepted in IEEE International Conference on Communications (IEEE ICC).  
# Authors: Li Yang (lyang339@uwo.ca) and Abdallah Shami (Abdallah.Shami@uwo.ca)  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# **Notebook 2: CNN Model Development**  
# Aims:  
# &nbsp; 1): Generate training and test images  
# &nbsp; 2): Construct CNN models (a CNN model by own, Xception, VGG16, VGG19, Resnet, Inception, InceptionResnet)  
# &nbsp; 3): Tune the hyperparameters of CNN models (hyperparameter optimization)  

# ## Import libraries

from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials
from PIL import Image
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import  ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import  Xception
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import plot_model
import datetime
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import sklearn.metrics as metrics
import statistics 
import tensorflow.keras as keras
import tensorflow.keras.callbacks as kcallbacks
import time

# ### Define the image plotting functions

#generate training and test images
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=128	#could try 128 or 32

# ## Generate Training and Test Images

# Normalization

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
# num_class = len(os.listdir('./test_224'))

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

history_this= LossHistory()

# ### Define the processing time measurement functions

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

timer = TimeMeasurement()

# ### Define the output sheet

class output_sheet:
    def __init__(self, columns:list=['accuracy', 'loss', 'val_acc', 'val_loss', 'hpo_time', 'train_time']):
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
        if path is None: path='2-result-{}.xlsx'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
        # self.apply_index()
        self.output_df.to_excel(path)

# ### Model 1: a CNN model by own (baseline)

def cnn_by_own(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,input_shape=INPUT_SIZE,num_class=num_class,epochs=20,patience=2, dropout_rate=0.5,verbose=0,savepath='./model_own.h5'):
    model = Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=input_shape,padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='glorot_uniform'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(num_class,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #train model
    earlyStopping=kcallbacks.EarlyStopping(monitor='val_acc', patience=patience, verbose=verbose, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_acc', verbose=verbose, save_best_only=True, mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        verbose=verbose
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 2: Xception

def xception(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class,epochs=20,frozen=131,lr=0.001,patience=2, dropout_rate=0.5,verbose=0,savepath='./xception.h5',input_shape=INPUT_SIZE):
    model_fine_tune = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:		#could be tuned to be 50, 100, or 131
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='xception')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#patience could be tuned by 2 and 3
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        #use_multiprocessing=True, 
        verbose=verbose,
        callbacks=callbacks,
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 3: VGG16

def vgg16(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class,epochs=20,frozen=15,lr=0.001,patience=2, dropout_rate=0.5,verbose=0, savepath='./VGG16.h5',input_shape=INPUT_SIZE):
    model_fine_tune = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:	#the number of frozen layers for transfer learning, have tuned from 5-18
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='vgg')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	#tuned learning rate to be 0.001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	#set the loss function to be binary crossentropy
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#set early stop patience to save training time
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        #use_multiprocessing=True, 
        #workers=2,
        callbacks=callbacks,
        verbose = verbose
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 4: VGG19

def vgg19(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class,epochs=20,frozen=19,lr=0.001,patience=2, dropout_rate=0.5,verbose=0,savepath='./VGG19.h5',input_shape=INPUT_SIZE):
    model_fine_tune = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:	#the number of frozen layers for transfer learning, have tuned from 5-18
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='vgg')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	#tuned learning rate to be 0.001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	#set the loss function to be binary crossentropy
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#set early stop patience to save training time
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=verbose,
        #use_multiprocessing=True, 
        #workers=2,
        callbacks=callbacks,
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 5: ResNet

def resnet(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class, epochs=20,frozen=120,lr=0.001,patience=2, dropout_rate=0.5,verbose=0,savepath='./resnet.h5',input_shape=INPUT_SIZE):
    model_fine_tune = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:	#the number of frozen layers for transfer learning, have tuned from 50-150
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:	#the number of trainable layers for transfer learning
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='resnet')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	#tuned learning rate to be 0.001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #set the loss function to be binary crossentropy
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#set early stop patience to save training time
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        #use_multiprocessing=True, 
        verbose=verbose,
        callbacks=callbacks,
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 6: Inception

def inception(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class, epochs=20,frozen=35,lr=0.001,patience=2, dropout_rate=0.5,verbose=0,savepath='./inception.h5',input_shape=INPUT_SIZE):
    model_fine_tune = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:	#the number of frozen layers for transfer learning, have tuned from 50-150
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:	#the number of trainable layers for transfer learning
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='resnet')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	#tuned learning rate to be 0.001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #set the loss function to be binary crossentropy
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#set early stop patience to save training time
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        #use_multiprocessing=True, 
        verbose=verbose,
        callbacks=callbacks,
    )
    return (history.get_best(), timer.get_processing_time())

# ### Model 7: InceptionResnet

def inceptionresnet(train_generator=train_generator,validation_generator=validation_generator,history:LossHistory=history_this,timer:TimeMeasurement=timer,num_class=num_class, epochs=20,frozen=500,lr=0.001,patience=2, dropout_rate=0.5,verbose=0,savepath='./inceptionresnet.h5',input_shape=INPUT_SIZE):
    model_fine_tune = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in model_fine_tune.layers[:frozen]:	#the number of frozen layers for transfer learning, have tuned from 400-550
        layer.trainable = False
    for layer in model_fine_tune.layers[frozen:]:	#the number of trainable layers for transfer learning
        layer.trainable = True
    model = GlobalAveragePooling2D()(model_fine_tune.output)
    model=Dense(units=256,activation='relu')(model)
    model=Dropout(dropout_rate)(model)
    model = Dense(num_class, activation='softmax')(model)
    model = Model(model_fine_tune.input, model, name='resnet')
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	#tuned learning rate to be 0.001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #set the loss function to be binary crossentropy
    #train model
    earlyStopping = kcallbacks.EarlyStopping(
        monitor='val_acc', patience=patience, verbose=verbose, mode='auto')	#set early stop patience to save training time
    saveBestModel = kcallbacks.ModelCheckpoint(
        filepath=savepath,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        mode='auto')
    callbacks=[earlyStopping,saveBestModel,history,timer]
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        #use_multiprocessing=True, 
        verbose=verbose,
        callbacks=callbacks,
    )
    return (history.get_best(), timer.get_processing_time())

# # Hyperparameter Optimization 
# 
# Hyperparameter optimization methods:
# 1. Random search
# 2. Bayesian optimization - Tree Parzen Estimator(BO-TPE)

def prediction(model, rootdir='./test_224/'):
#read images from validation folder
    test_laels = []
    test_images=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not (file.endswith(".jpeg"))|(file.endswith(".jpg"))|(file.endswith(".png")):
                continue
            test_laels.append(subdir.split('/')[-1])
            test_images.append(os.path.join(subdir, file))

    predict=[]
    length=len(test_images)
    label=validation_generator.class_indices
    label={v: k for k, v in label.items()}
    for i in range(length):
        inputimg=test_images[i]
        test_batch=[]
        thisimg=np.array(Image.open(inputimg))/255 #read all the images in validation set
        #print(thisimg)
        test_shape=(1,)+thisimg.shape
        thisimg=thisimg.reshape(test_shape)
        model_batch=model.predict(thisimg) #use master model to process the input image
        #generate result by model 1
        prob=model_batch[0,np.argmax(model_batch,axis=1)[0]]
        res=label[np.argmax(model_batch,axis=1)[0]]
        predict.append(res)
    acc=accuracy_score(test_laels,predict)
    return acc

#define the objective function to be optimized

# ## Model by Own

def objective_model_own(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    cnn_by_own(input_shape=INPUT_SIZE, train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./model_own.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## Xception

#define the objective function to be optimized

def objective_xception(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    xception(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./xception.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## VGG16

#define the objective function to be optimized

def objective_vgg16(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    vgg16(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./VGG16.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## VGG19

#define the objective function to be optimized

def objective_vgg19(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    vgg19(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./VGG19.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## ResNet

#define the objective function to be optimized

def objective_resnet(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    resnet(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./resnet.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## Inception

#define the objective function to be optimized

def objective_inception(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    inception(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./inception.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# ## InceptionResnet

#define the objective function to be optimized

def objective_inceptionresnet(params, num_class=num_class, train_generator=train_generator, validation_generator=validation_generator):
    
    params = {
        'frozen': int(params['frozen']),
        'epochs': int(params['epochs']),
        'patience': int(params['patience']),
        'lr': abs(float(params['lr'])),
        'dropout_rate': abs(float(params['dropout_rate'])),
    }

    inceptionresnet(train_generator=train_generator, validation_generator=validation_generator, num_class=num_class, **params)

    acc=prediction(model=load_model('./inceptionresnet.h5'))

    print('accuracy:%s'%acc)
    return {'loss': -acc, 'status': STATUS_OK }

# Multiprocessing helper method
# This is aimed to solved the OOM issue

def run_with_multiprocessing(func, **kwds):
    pool = multiprocessing.Pool(processes=1)
    result = pool.apply_async(func=func, kwds=kwds)
    pool.close()
    pool.join()
    return result.get()

if __name__ == '__main__':

    starting_time = time.time()

    # Prepare output_sheet
    output = output_sheet(columns=['accuracy', 'loss', 'val_acc', 'val_loss', 'hpo_time', 'train_time'])

    # # Construct CNN models

    # ### Model 1: a CNN model by own (baseline)

    best_result, processing_time = run_with_multiprocessing(func=cnn_by_own, input_shape=INPUT_SIZE,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # cnn_by_own(input_shape=INPUT_SIZE,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('model_own', train_time=processing_time, **best_result)

    # ### Model 2: Xception

    #default only 50, tf36cnn 99
    best_result, processing_time = run_with_multiprocessing(func=xception,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # xception(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('Xception', train_time=processing_time, **best_result)

    # ### Model 3: VGG16

    best_result, processing_time = run_with_multiprocessing(func=vgg16,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # vgg16(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)	#tf36cnn
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('VGG16', train_time=processing_time, **best_result)

    # ### Model 4: VGG19

    best_result, processing_time = run_with_multiprocessing(func=vgg19,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # vgg19(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)	#binary classificaiton
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('VGG19', train_time=processing_time, **best_result)

    # ### Model 5: ResNet

    best_result, processing_time = run_with_multiprocessing(func=resnet,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # resnet(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)	#binary classificaiton
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('Resnet', train_time=processing_time, **best_result)

    # ### Model 6: Inception

    best_result, processing_time = run_with_multiprocessing(func=inception,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # inception(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)	#binary classificaiton
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('Inception', train_time=processing_time, **best_result)

    # ### Model 7: InceptionResnet

    best_result, processing_time = run_with_multiprocessing(func=inceptionresnet,num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)
    # inceptionresnet(num_class=num_class,epochs=20,verbose=1,history=history_this,timer=timer)	# 5-class classificaiton
    # history_this.loss_plot('epoch')
    # history_this.loss_plot('batch')
    # plt.show()

    output.add('InceptionResnet', train_time=processing_time, **best_result)

    # # Hyperparameter Optimization 

    # ## Model by Own

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_model_own,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_model_own,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=cnn_by_own, input_shape=INPUT_SIZE, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # cnn_by_own(input_shape=INPUT_SIZE, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('model_own (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_model_own,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_model_own,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=cnn_by_own, input_shape=INPUT_SIZE, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # cnn_by_own(input_shape=INPUT_SIZE, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('model_own (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## Xception

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    available_frozen = [50, 100, 131]
    space = {
        'frozen': hp.choice('frozen', available_frozen),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_xception,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_xception,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': available_frozen[int(best['frozen'])],
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=xception, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # xception(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('Xception (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    #Hyperparameter optimization by Random search
    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_xception,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_xception,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': available_frozen[int(best['frozen'])],
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=xception, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # xception(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('Xception (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## VGG16

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'frozen': hp.quniform('frozen', 15, 18, 1),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_vgg16,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_vgg16,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=vgg16, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # vgg16(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('VGG16 (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    #Hyperparameter optimization by Random search
    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_vgg16,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_vgg16,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=vgg16, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # vgg16(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('VGG16 (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## VGG19

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'frozen': hp.quniform('frozen', 15, 18, 1),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_vgg19,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_vgg19,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=vgg19, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # vgg19(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('VGG19 (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    #Hyperparameter optimization by Random search

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_vgg19,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_vgg19,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=vgg19, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # vgg19(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('VGG19 (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## ResNet

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'frozen': hp.quniform('frozen', 50, 150, 10),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_resnet,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_resnet,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=resnet, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # resnet(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('ResNet (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    #Hyperparameter optimization by Random search

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_resnet,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_resnet,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=resnet, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # resnet(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('ResNet (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## Inception

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'frozen': hp.quniform('frozen', 50, 150, 10),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_inception,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_inception,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=inception, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # inception(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('Inception (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    #Hyperparameter optimization by Random search

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_inception,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_inception,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=inception, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # inception(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('Inception (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ## InceptionResnet

    # ### BO-TPE

    #Hyperparameter optimization by Bayesian optimization - Tree Parzen Estimator
    space = {
        'frozen': hp.quniform('frozen', 400, 500, 10),
        'epochs': hp.quniform('epochs', 5, 21, 5),
        'patience': hp.quniform('patience', 2, 4, 1),
        'lr': hp.quniform('lr', 0.001, 0.006, 0.001),
        'dropout_rate': hp.quniform('dropout_rate', 0.3, 0.6, 0.1),
    }

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_inceptionresnet,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # best = fmin(fn=objective_inceptionresnet,
    #             space=space,
    #             algo=tpe.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=inceptionresnet, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # inceptionresnet(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('InceptionResnet (BO-TPE)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # ### Random Search

    #Hyperparameter optimization by Random search

    t1=time.time()
    best = run_with_multiprocessing(func=fmin,
                fn=objective_inceptionresnet,
                space=space,
                algo=rand.suggest,
                max_evals=10)
    # best = fmin(fn=objective_inceptionresnet,
    #             space=space,
    #             algo=rand.suggest,
    #             max_evals=10)
    print("Hyperopt estimated optimum {}".format(best))
    t2=time.time()
    print("Time: "+str(t2-t1))

    # Retrain the model by using the best hyperparameter values to obtain the best model
    params = {
            'frozen': int(best['frozen']),
            'epochs': int(best['epochs']),
            'patience': int(best['patience']),
            'lr': abs(float(best['lr'])),
            'dropout_rate': abs(float(best['dropout_rate'])),
        }
    best_result, processing_time = run_with_multiprocessing(func=inceptionresnet, num_class=num_class, verbose=1, history=history_this, timer=timer, **params)
    # inceptionresnet(num_class=num_class, verbose=1, history=history_this, timer=timer, **params)

    output.add('InceptionResnet (Random Search)', hpo_time=t2-t1, train_time=processing_time, **best_result)

    # # Save Result

    output.to_excel()

    # Online GPU renting platform specification
    # WeChat Message
    import requests
    resp = requests.get(
        "https://www.autodl.com/api/v1/wechat/message/push?token={token}&title={title}&name={name}&content={content}".format(
            token="add7ba8eea4b",
            title="Running Completed",
            name="2-CNN_Model_Development&Hyperparameter Optimization.py",
            content="Time Used: {}".format(time.time()-starting_time))
    )
    print(resp.content.decode())


