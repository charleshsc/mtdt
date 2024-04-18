import d4rl
from ast import parse
import numpy as np
import torch
import wandb
import os
import time
import pathlib
from prompt_dt.fl_utils import * 
import argparse
import pickle
import random
import sys

import itertools
import copy
from tqdm import trange

from prompt_dt.prompt_decision_transformer import PromptDecisionTransformer
from prompt_dt.seq_trainer import SequenceTrainer
from prompt_dt.taming_trainer import TamingTrainer
from prompt_dt.prompt_utils import get_env_list, report_parameters
from prompt_dt.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from prompt_dt.prompt_utils import process_total_data_mean, load_data_prompt, process_info, load_meta_data_prompt
from prompt_dt.prompt_utils import eval_episodes, finetune_hf_episodes, finetune_hf_episodes_offline
from prompt_dt.fl_utils import ERK_maskinit

from prompt_dt.PCGrad import PCGrad, CAGrad

from collections import namedtuple
import json, pickle, os
from logger import setup_logger, logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

mt50_task_list = [
    'basketball-v2', 'bin-picking-v2', 'button-press-topdown-v2','button-press-v2', 
    'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 
    'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
    'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 
    'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2',  'handle-press-side-v2', 
    'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 
    'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 
    'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 
    'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2',  'soccer-v2', 
    'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 
    'window-open-v2', 'window-close-v2','assembly-v2','button-press-topdown-wall-v2',
    'hammer-v2','peg-unplug-side-v2', 'reach-wall-v2', 'stick-push-v2', 
    'stick-pull-v2', 'box-close-v2'
    ]
mt5_task_list = ['basketball-v2', 'bin-picking-v2', 'button-press-topdown-v2',
    'button-press-v2', 'button-press-wall-v2',]
mt30_task_list = ['basketball-v2', 'bin-picking-v2', 'button-press-topdown-v2',
    'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
    'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
    'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
    'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
    'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2',]

mt250_task_list = [s+f'-{i}' for s in mt5_task_list for i in range(5)]


def experiment_mix_env(
        exp_prefix,
        variant,
):
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']
    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']
    HF = variant['with_hf']
    Evaluation = variant['evaluation']
    hf_offline = variant['hf_offline']
    seed = variant['seed']
    set_seed(variant['seed'])
    m = variant['m']
    smooth = variant['smooth']
    env_name_ = variant['env']
    
    ######
    # construct train and test environments
    ######
    eta_min=0
    eta_max=variant["mask_change_max"]
    cur_dir = os.getcwd()
    data_save_path = 'MT250'
    # data_save_path = 'MT50/dataset'
    save_path = variant['save_path']
    timestr = time.strftime("%y%m%d-%H%M%S")
    
    # if '50' in variant['prefix_name']:
    #     train_env_name_list, test_env_name_list = mt50_task_list, mt50_task_list
    # elif '30' in variant['prefix_name']:
    #     train_env_name_list, test_env_name_list = mt30_task_list, mt30_task_list
    # elif '5' in variant['prefix_name']:
    #     train_env_name_list, test_env_name_list = mt5_task_list, mt5_task_list
    # else:
    #     raise NotImplementedError("Please use the 50,30,5")

    
    train_env_name_list, test_env_name_list = mt10_task_list, mt10_task_list


    # training envs
    info, _ = get_env_list(train_env_name_list, device, total_env='metaworld', seed=seed)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, device,total_env='metaworld_test', seed=seed)

    num_env = len(train_env_name_list)
    exp_prefix = '-'.join(train_env_name_list)
    group_name = variant['prefix_name']
    Evaluation_token = 'Evaluation' if Evaluation else '-'
    n_layer=variant['n_layer']
    n_head=variant['n_head']
    exp_prefix = f'{Evaluation_token}-{seed}-{timestr}-{n_layer}l{n_head}h'
    if variant['no_prompt']:
        exp_prefix += '_NO_PROMPT'
    if variant['finetune']:
        exp_prefix += '_FINETUNE'
    if variant['no_r']:
        exp_prefix += '_NO_R'
    if variant['suboptimal']:
        exp_prefix += '_SUBOPTIMAL'
    save_path = os.path.join(save_path, group_name+'-'+exp_prefix)
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    setup_logger(exp_prefix, variant=variant, log_dir=save_path)

    logger.log(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    
    ######
    # process train and test datasets
    ######
    # load training dataset
    if variant['suboptimal'] is True:
        flag=False
    else:
        flag=True
    
    trajectories_list, prompt_trajectories_list = load_meta_data_prompt(
        train_env_name_list, 
        data_save_path, 
        flag,
    )

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant, logger)
    prompt_info = copy.deepcopy(info)
    # process test info
    test_info = process_info(test_env_name_list, trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant, logger)
    prompt_test_info = copy.deepcopy(test_info)

    ######
    # construct dt model and trainer
    ######

    # state_dim = test_env_list[0].observation_space.shape[0]
    # act_dim = test_env_list[0].action_space.shape[0]

    model = PromptDecisionTransformer(
        env_name_list=train_env_name_list,
        info=info,
        special_embedding=variant['special_embedding'],
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)
    report_parameters(model, logger)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name_list[0]
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch(trajectories_list[0], info[env_name], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories_list[0], prompt_info[env_name], variant),
        get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list),
        logger=logger,
        variant=variant
    )

    if not variant['evaluation']:
        ######
        # start training
        ######

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        
        best_ret = -10000
        best_iter = 0

        for iter in trange(variant['max_iters']):
            env_id = iter % num_env
            env_name = train_env_name_list[env_id]

            logs, _ = trainer.pure_train_iteration_mix(
                num_steps=2, 
                no_prompt=args.no_prompt,
                masks = None,
                env_name = env_name,
            )
            
            
            if (iter+1) % args.test_eval_interval == 0:   
                
                # log training information
                logger.record_tabular('Env Name', env_name) 
                for key, value in logs.items():
                    logger.record_tabular(key, value)
                logger.dump_tabular() 

                group='test'
                
                # evaluate test
                test_eval_logs = trainer.eval_iteration_metaworld(
                    None, get_prompt, prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, test_info, variant, test_env_list, iter_num=iter + 1, 
                    no_prompt=args.no_prompt, group=group)

                total_success_mean = test_eval_logs[f'{group}-Total-Success-Mean']
                if total_success_mean > best_ret:
                    best_ret = total_success_mean
                    best_iter = iter + 1
                
                logger.log('Best success: {}, Iteration {}'.format(best_ret, best_iter))
            
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter + 1),  folder=save_path, env_masks=None)

    else:
        ####
        # start evaluating
        ####
        saved_model_path = variant['load_path']
        model_dict = {}
        for i, path in enumerate(saved_model_path):
            new_model_dict = torch.load(path)
            for key, value in new_model_dict.items():
                if key in model_dict.keys():
                    model_dict[key] = (model_dict[key] * i + value) / (i+1)
                else:
                    model_dict[key] = value
        
        model.load_state_dict(model_dict)
        logger.log('model initialized from: ' + ' '.join(saved_model_path))
        # eval_iter_num = int(saved_model_path.split('_')[-1])

        
        eval_logs = trainer.eval_iteration_multienv(
                get_prompt, prompt_trajectories_list,
                eval_episodes, test_env_name_list, test_info, test_info, variant, test_env_list, iter_num=0, 
                print_logs=True, no_prompt=args.no_prompt, group='eval')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default='hopper') 
    parser.add_argument('--dataset_mode', type=str, default='medium')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--test_dataset_mode', type=str, default='medium')
    parser.add_argument('--train_prompt_mode', type=str, default='medium')
    parser.add_argument('--test_prompt_mode', type=str, default='medium')
    parser.add_argument('--name', type=str, default='gym-experiment')

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=False)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--average_state_mean', action='store_true', default=False) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load_path', type=str, nargs='+', default=None) # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=5) 
    parser.add_argument('--num_finetune_episodes', type=int, default=2)
    parser.add_argument('--max_iters', type=int, default=5000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', action='store_true', default=False)
    parser.add_argument('--train_eval_interval', type=int, default=50)
    parser.add_argument('--mask_interval', type=int, default=10)
    parser.add_argument('--test_eval_interval', type=int, default=1000)
    parser.add_argument('--test_eval_seperate_interval', type=int, default=5000)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--save_path', type=str, default='./save/')

    parser.add_argument('--special_embedding', action='store_true', default=False)
    parser.add_argument('--prefix_name', type=str, default='MT50')
    parser.add_argument('--suboptimal', action='store_true', default=False)

    args = parser.parse_args()
    
    experiment_mix_env(args.name, variant=vars(args))