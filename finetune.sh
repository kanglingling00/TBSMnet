# Finetune on KITTI 2015
## max_disp = 192             
python finetune.py --max_disp=192 \
                --datapath='data'\
                --resume_model='log/TBSMnet_192/TBSMnet_192_scenceflow_kitti_epoch10.pth' 