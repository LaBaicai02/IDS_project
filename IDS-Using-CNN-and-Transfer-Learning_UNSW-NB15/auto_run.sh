#!/bin/bash

count=0
while [ $count -lt 3 ]
do
    count=`expr $count + 1`
    echo "Starting round NO.${count}"
    echo "Starting running 2-CNN_Model_Development"
    python 2-CNN_Model_Development\&Hyperparameter\ Optimization.py
    echo "Starting running 3-Ensemble_Models_CAN"
    python 3-Ensemble_Models-CAN.py
    echo "Ending round NO.${count}"
done
curl -G -d 'token=' --data-urlencode 'title=Processing Finished' --data-urlencode 'name=CNN' --data-urlencode 'content=Complete' https://www.autodl.com/api/v1/wechat/message/push
shutdown
