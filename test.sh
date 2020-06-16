#!/bin/sh
source ~/pytorch_env/bin/activate
#python test_widerface.py --trained_model ./train_weights/Resnet50_epoch_40.pth --network resnet50 --dataset_folder /data/backup/widerface_retinaface_gt_v1.1/val/images/ --save_image
python test_dir.py --trained_model ./train_weights/Resnet50_Final.pth --network resnet50 --dataset_folder /home/alex/data/300W/01_Indoor --save_image

