#!/bin/sh
source ~/pytorch_env/bin/activate

time_str=`date +%Y%m%d_%H%M%S`
log_file="logs/train_retinaface_$time_str.log"

#nohup python train.py --network resnet50 --training_dataset /data/backup/widerface_retinaface_gt_v1.1/train/label.txt --save_folder ./train_weights > $log_file 2>&1 &
nohup python train.py --network resnet50 --training_dataset /home/alex/data/300W/01_Indoor --save_folder ./train_weights > $log_file 2>&1 &

