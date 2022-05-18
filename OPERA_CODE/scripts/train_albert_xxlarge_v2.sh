CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --batch_size 128 --max_epoch 12 --TMSPAN --pretrain_model albert.xxlarge --gradient_accumulation_steps 32 --add_op_layer --warmup_schedule warmup_cosine --SEED 678 --LR 1e-4 --BLR 3e-5 --data_root OPERA_DATA

