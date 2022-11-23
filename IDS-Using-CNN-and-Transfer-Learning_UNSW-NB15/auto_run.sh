#!/bin/bash

# Token for messaging (should be replace by your own)
token=123

# Config of messaging platform
interface="https://www.autodl.com/api/v1/wechat/message/push"

# How many sets of data do you need
turn=3

count=0
while [ $count -lt $turn ]
do
    count=`expr $count + 1`
    echo "Starting round NO.${count}"

    # # Clean dir
    # rm -r test test_224 train train_224

    # # Code 1
    # echo "Starting running 1-Data_pre-processing_CAN"
    # python 1-Data_pre-processing_CAN.py
    # # Complete Code 2 Messaging
    # curl -G -d 'token='$token --data-urlencode 'title=Processing Finished' --data-urlencode 'name=CNN_Based Code 1' --data-urlencode 'content=Complete' $interface
    
    # Code 2
    echo "Starting running 2-CNN_Model_Development"
    python 2-CNN_Model_Development\&Hyperparameter\ Optimization.py
    # Complete Code 2 Messaging
    curl -G -d 'token='$token --data-urlencode 'title=Processing Finished' --data-urlencode 'name=CNN_Based Code 2' --data-urlencode 'content=Complete' $interface
    
    # Code 3
    echo "Starting running 3-Ensemble_Models_CAN"
    python 3-Ensemble_Models-CAN.py
    # Complete Messaging
    curl -G -d 'token='$token --data-urlencode 'title=Processing Finished' --data-urlencode 'name=CNN_Based Code 3' --data-urlencode 'content=Complete' $interface

    echo "Ending round NO.${count}"
done

# Complete Messaging
curl -G -d 'token='$token --data-urlencode 'title=Processing Finished' --data-urlencode 'name=CNN_Based' --data-urlencode 'content=Complete' $interface

# Auto Shutdown
shutdown
