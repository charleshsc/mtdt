CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 1000 --n_layer 6 --n_head 8 \
    --test_eval_interval 10  --num_eval_episodes 10 \
    --mask_interval 5  --prefix_name unseen --env cheetah_dir \
    --data_path ./MT50/unseen/data

CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 1000 --n_layer 6 --n_head 8 \
    --test_eval_interval 10  --num_eval_episodes 10 \
    --mask_interval 5  --prefix_name unseen --env cheetah_vel \
    --data_path ./MT50/unseen/data

CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 1000 --n_layer 6 --n_head 8 \
    --test_eval_interval 10  --num_eval_episodes 10 \
    --mask_interval 5  --prefix_name unseen --env ant_dir \
    --data_path ./MT50/unseen/data

CUDA_VISIBLE_DEVICES=$1 python pdt_meta.py --seed 123 --max_iters 1000 --n_layer 6 --n_head 8 \
    --test_eval_interval 10  --num_eval_episodes 10 \
    --mask_interval 5  --prefix_name unseen --env ML1-pick-place-v2 \
    --data_path ./MT50/unseen/data