torchrun --standalone --nproc_per_node=1 train.py \
  --image_dir /home/djhuai/zuo/mobicom/dggt_2/dggt/data/nuscenes/processed_10Hz/mini \
  --ckpt_path /home/djhuai/zuo/mobicom/dggt_2/pretrained/model_latest_waymo.pt \
  --log_dir logs/nuscenes_dynamic \
  --sequence_length 4 \
  --covisibility_sequence_length 4

把 --ckpt_path 替换成实际预训练权重路径。若使用 6 张 GPU，将 --nproc_per_node=8 改成 6。

注意：当前 train.py 中场景列表固定为 300 到 599，但你的数据目录示例是 000 到 010。启动前需要确认这与数据集场景编号相符，否则数据集可能找不到对应场景。