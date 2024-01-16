import numpy as np
import gym
import json, pickle, random, os, torch
import metaworld
from collections import namedtuple
from .prompt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg

""" constructing envs """

def gen_env(env_name, seed=1, total_env=None, num_eval_episodes=0):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
        dversion = 2
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
        dversion = 2
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
        dversion = 2
    elif env_name == 'reacher2d':
        # from decision_transformer.envs.reacher_2d import Reacher2dEnv
        # env = Reacher2dEnv()
        env = gym.make('Reacher-v4')
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
        dversion = 2
    elif env_name == 'pen':
        env = gym.make('pen-v0')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
        dversion = 1
    elif env_name == 'hammer':
        # env = gym.make('hammer-human-longhorizon-v0')
        env = gym.make('hammer-v0')
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.
        dversion = 1
    elif env_name == 'door':
        env = gym.make('door-v0')
        max_ep_len = 1000
        env_targets = [2000, 1000]
        scale = 100.
        dversion = 1
    elif env_name == 'relocate':
        env = gym.make('relocate-v0')
        max_ep_len = 1000
        env_targets = [3000, 1000]
        scale = 1000.
        dversion = 1
    elif env_name == 'kitchen':
        env = gym.make('kitchen-complete-v0')
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.
        dversion = 0
    else:
        if 'metaworld' in total_env:
            task = metaworld.MT1(env_name).train_tasks[0]
            env = metaworld.MT1(env_name).train_classes[env_name]()
            env.set_task(task)
            env.seed(seed)
            max_ep_len = 500
            env_targets = [4500]
            scale = 1000.
            dversion = 0 #compatible

            if 'test' in total_env:
                task = [metaworld.MT1(env_name).train_tasks[i] for i in range(num_eval_episodes)]
                mt1 = [metaworld.MT1(env_name) for i in range(num_eval_episodes)]
                env_list = [mt1[i].train_classes[env_name]() for i in range(num_eval_episodes)]
                for i in range(len(env_list)):
                    env_list[i].set_task(task[i])
                    env_list[i].seed(seed)
                env = env_list

        else:
            raise NotImplementedError
    return env, max_ep_len, env_targets, scale, dversion


def get_env_list(env_name_list, device, total_env=None, seed=1, num_eval_episodes=10):
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale, dversion = gen_env(env_name=env_name, seed=seed, total_env=total_env, num_eval_episodes=num_eval_episodes)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        if type(env) is list:
            info[env_name]['state_dim'] = env[0].observation_space.shape[0]
            info[env_name]['act_dim'] = env[0].action_space.shape[0] 
        else:
            info[env_name]['state_dim'] = env.observation_space.shape[0]
            info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        info[env_name]['dversion'] = dversion
        env_list.append(env)
    return info, env_list




""" prompts """

def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1)) 
    return [p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask]


def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1, index=None):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes*sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )
        assert len(prompt_trajectories) == len(sorted_inds)
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes*sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                if i+1 > len(sorted_inds):
                    i = i % len(sorted_inds)
                traj = prompt_trajectories[int(sorted_inds[(i+1)])] 
            
            if index is not None:
                traj = prompt_trajectories[int(sorted_inds[index])]

            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, prompt_info, variant, train_env_name_list):
    per_env_batch_size = variant['batch_size']

    def fn(batch_size=per_env_batch_size, index=None):
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        # env_id = index % len(train_env_name_list)
        env_id = train_env_name_list.index(index)
        env_name = index
        if prompt_trajectories_list:
            get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
        else:
            get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)
        get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) 
        prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size)
        p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
        p_s_list.append(p_s)
        p_a_list.append(p_a)
        p_r_list.append(p_r)
        p_d_list.append(p_d)
        p_rtg_list.append(p_rtg)
        p_timesteps_list.append(p_timesteps)
        p_mask_list.append(p_mask)

        batch = get_batch_fn(batch_size=batch_size)
        s, a, r, d, rtg, timesteps, mask = batch
        if variant['no_r']:
            r = torch.zeros_like(r)
        if variant['no_rtg']:
            rtg = torch.zeros_like(rtg)
        s_list.append(s)
        a_list.append(a)
        r_list.append(r)
        d_list.append(d)
        rtg_list.append(rtg)
        timesteps_list.append(timesteps)
        mask_list.append(mask)

        # p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        # p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        # s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        # rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask, env_name
        return prompt, batch
    return fn

""" batches """

def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['K']

    def fn(batch_size=batch_size, max_len=K, start=0, stochastic=True):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_finetune(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K = variant['batch_size'], variant['prompt_length'] # use the same amount of data for funetuning

    def fn(batch_size=batch_size, max_len=K, start=0, stochastic=True):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(start, start + batch_size):
            if stochastic:
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
            else:
                if i > len(sorted_inds):
                    i = 1
                traj = trajectories[int(sorted_inds[-i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn

""" data processing """

def process_total_data_mean(trajectories, mode):

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return state_mean, state_std


def process_dataset(trajectories, mode, env_name, dataset, pct_traj, logger):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    logger.log('=' * 50)
    logger.log(f'Starting new experiment: {env_name} {dataset}')
    logger.log(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logger.log(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logger.log(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logger.log('=' * 50)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_data_prompt(env_name_list, data_save_path, dataset, prompt_mode, info):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:
        dversion = info[env_name]['dversion']
        dataset_path = data_save_path+f'/{env_name}-{dataset}-v{dversion}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        trajectories_list.append(trajectories)
        
        prompt_dataset_path = data_save_path+f'/{env_name}-{dataset}-v{dversion}.pkl'
        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        prompt_trajectories_list.append(prompt_trajectories)

    return trajectories_list, prompt_trajectories_list

def load_meta_data_prompt(env_name_list, data_save_path, optimal=True):
    trajectories_list = []
    prompt_trajectories_list = []

    length = 2000 if optimal else 1000
    for task_id in range(len(env_name_list)):
        path = os.path.join(data_save_path, env_name_list[task_id])
        cur_task_trajs = []
        for i in range(length-2):
            cur_path = os.path.join(path, f"{i}.npz")
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_trajs.append(episode)
        trajectories_list.append(cur_task_trajs)

        cur_task_prompt_trajs = []
        for i in range(length-2, length):
            cur_path = os.path.join(path, f"{i}.npz")
            with open(cur_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
            cur_task_prompt_trajs.append(episode)
        prompt_trajectories_list.append(cur_task_trajs)
    
    return trajectories_list, prompt_trajectories_list
                


def process_info(env_name_list, trajectories_list, info, mode, dataset, pct_traj, variant, logger):
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], dataset=dataset, pct_traj=pct_traj, logger=logger)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std
        if variant['average_state_mean']:
            info[env_name]['state_mean'] = variant['total_state_mean']
            info[env_name]['state_std'] = variant['total_state_std']
    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, info, prompt=None):
        returns = []
        success = []
        length = []
        for i in range(num_eval_episodes):
            with torch.no_grad():
                ret, lens, suc = prompt_evaluate_episode_rtg(
                    env_name,
                    env[i],
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize'],
                    info=info,               
                    )
            returns.append(ret)
            length.append(lens)
            success.append(suc)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            # f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            f'{env_name}_target_{target_rew}_length_mean': np.mean(length),
            # f'{env_name}_target_{target_rew}_length_std': np.std(length),
            f'{env_name}_target_{target_rew}_success_mean': np.mean(success),
            # f'{env_name}_target_{target_rew}_success_std': np.std(success),
            }
    return fn

def finetune_hf_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_finetune_episodes = variant['num_finetune_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, prompt=None):
        returns = []
        for _ in range(num_finetune_episodes):
            with torch.no_grad():
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
        return np.mean(returns)
    return fn

def finetune_hf_episodes_offline(get_batch, variant, loss_fn):
    num_finetune_episodes = variant['num_finetune_episodes']
    batch_size = variant['finetune_batch_size']

    def fn(model, prompt=None, start=0):
        returns = []
        for _ in range(num_finetune_episodes):
            with torch.no_grad():
                states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(batch_size, start=start, stochastic=True)
                action_target = torch.clone(actions)

                state_preds, action_preds, reward_preds = model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
                )

                act_dim = action_preds.shape[2]
                action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

                loss = loss_fn(
                    None, action_preds, None,
                    None, action_target, None,
                )

            returns.append(loss.detach().cpu().item())
        return np.mean(returns)
    return fn

def _to_str(num):
    if num >= 1e6:
        return f'{(num/1e6):.2f} M'
    else:
        return f'{(num/1e3):.2f} k'

def param_to_module(param):
    module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
    return module_name

def report_parameters(model, logger, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    logger.log(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        logger.log(' '*8 + f'{key:10}: {_to_str(count)} | {modules[module]}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    logger.log(' '*8 + f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters