CUDA_VISIBLE_DEVICES=$1 python mtdt_meta.py --seed 123 --max_iters 5000000 --n_layer 9 --n_head 16 \
    --test_eval_interval 50000  --num_eval_episodes 10 \
    --prefix_name mt125 --no-prompt  --save_path ./MT250_save/