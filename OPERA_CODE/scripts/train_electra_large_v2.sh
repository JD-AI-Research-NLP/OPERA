CUDA_VISIBLE_DEVICES=6,7 python3 train.py --batch_size 16 --max_epoch 15 --pretrain_model electra.large --gradient_accumulation_steps 1 --warmup_schedule warmup_cosine --add_op_layer --data_root OPERA_DATA

