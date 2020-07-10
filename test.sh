#!/bin/sh
source ~/pytorch_env/bin/activate

weight_file=./train_weights/Resnet50_landmark8_Final.pth
#dataset=/data/backup/widerface_retinaface_gt_v1.1/val/images/
#dataset=/home/alex/data/facial_landmark/300W/01_Indoor
dataset=/home/alex/data/facial_landmark/300w_merge_single_face/testset



#python test_widerface.py --trained_model $weight_file --network resnet50 --dataset_folder $dataset --save_image
python test_dir.py --trained_model $weight_file --network resnet50 --dataset_folder $dataset --save_image

