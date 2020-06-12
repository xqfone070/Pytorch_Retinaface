#!/bin/sh
source ~/pytorch_env/bin/activate
python test_widerface.py --trained_model ./train_weights/resultsResnet50_Final.pth --network resnet50 --dataset_folder /data/backup/widerface_retinaface_gt_v1.1/val/images/ --save_image
