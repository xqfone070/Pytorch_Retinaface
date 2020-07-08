#!/bin/sh
source ~/pytorch_env/bin/activate

time_str=`date +%Y%m%d_%H%M%S`
log_file="logs/train_retinaface_$time_str.log"


#dataset=/data/backup/widerface_retinaface_gt_v1.1/train/label.txt
dataset=/home/alex/data/facial_landmark/300w_merge_single_face/trainset
#dataset=/home/alex/data/facial_landmark/300W/01_Indoor
#dataset=/home/alex/data/facial_landmark/300w_merge_1.2/trainset/whole
#nohup python train.py --network resnet50 --training_dataset /data/backup/widerface_retinaface_gt_v1.1/train/label.txt --save_folder ./train_weights > $log_file 2>&1 &
nohup python train.py --network resnet50 --training_dataset $dataset --save_folder ./train_weights --lr 0.001 > $log_file 2>&1 &
tailf $log_file

