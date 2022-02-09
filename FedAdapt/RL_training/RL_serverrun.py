if __name__ == "__main__":
	import pickle
	import torch
	import tqdm
	import numpy as np

	import logging
	logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)

	import sys
	sys.path.append('../FedAdapt')
	from RLEnv import Env
	import config
	import utils
	import RLEnv
	import PPO
	import time

	time_server_start = time.perf_counter()

	if config.random:
		torch.manual_seed(config.random_seed)
		np.random.seed(config.random_seed)
		logger.info('Random seed: {}'.format(config.random_seed))

	# Creating environment
	env = Env(0, config.SERVER_ADDR, config.SERVER_PORT, config.CLIENTS_LIST, config.model_name, config.model_cfg, config.rl_b)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Creating PPO agent
	state_dim = env.state_dim
	action_dim = env.action_dim
	memory = PPO.Memory()
	ppo = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)

	# RL training
	logger.info('==> RL Training Start.')
	time_step = 0
	update_epoch = 1
	
	res = {}
	res['rewards'], res['maxtime'], res['actions'], res['std'] = [], [], [], []
	metrics_dict = dict()
	

	for i_episode in tqdm.tqdm(range(1, config.max_episodes+1)):
		###############
		####EPISODE####
		###############	
		time_start_episode = time.perf_counter()
		episode_dict = dict()

		done = False # Flag controling finish of one episode
		if i_episode == 1: # We run two times of initial state to get stable training time
			first = True
			state = env.reset(done, first)
		else:
			first = False
			state = env.reset(done, first)

		for t in range(config.max_timesteps):
			############
			####STEP####
			############
			time_start_step_server = time.perf_counter()
			logger.info('====================================>')
			time_step +=1
			action, action_mean, std = ppo.select_action(state, memory)
			state, reward, maxtime, done = env.step(action, done)
			logger.info('Current reward: ' + str(reward))
			logger.info('Current maxtime: ' + str(maxtime))

			# Saving reward and is_terminals:
			memory.rewards.append(reward)
			memory.is_terminals.append(done)

			
			# Update
			if time_step % config.update_timestep == 0:
				ppo.update(memory)
				logger.info('Agent has been updated: ' + str(update_epoch))
				if update_epoch > config.exploration_times:
					ppo.explore_decay(update_epoch - config.exploration_times)

				memory.clear_memory()
				time_step = 0
				update_epoch += 1

				# Record the results for each update epoch
				with open('RL_res.pkl','wb') as f:
					pickle.dump(res,f)

				# save the agent every updates
				torch.save(ppo.policy.state_dict(), './PPO.pth')
			

			res['rewards'].append(reward)
			res['maxtime'].append(maxtime)
			res['actions'].append((action, action_mean))
			res['std'].append(std)

			time_finish_step_server = time.perf_counter()
			##
			step_dict = dict()
			step_dict['split_layer'] = config.split_layer
			step_dict['server_step_time_total'] = time_finish_step_server - time_start_step_server
			step_dict['client_step_time_total'] = env.total_iterations_time
			step_dict['client_last_iteration_time'] = env.infer_state
			step_dict['maxtime_iteration'] = maxtime
			step_dict['client_baseline'] = env.baseline
			step_dict['rewards'] = reward
			step_dict['server_idle_time'] = env.server_idle_time
			step_dict['client_interstep_idle_time'] = env.step_client_interstep_idle_time
			step_dict['client_offloading_idle_time'] = env.step_client_offloading_idle_time
			step_dict['actions'] = action
			step_dict['action_mean'] = action_mean
			step_dict['std'] = std
			step_dict['state'] = state
			# Capture all of the metrics of step T into episodes dict
			episode_dict["step_"+str(t)] = step_dict
			##
			
			if done:
				break

			# stop when get control update epoch
			if update_epoch > 10:
				break

		time_finish_episode = time.perf_counter()
		episode_dict["episode_time_total"] = time_finish_episode - time_start_episode
		metrics_dict["episode_" + str(i_episode)] = episode_dict
		metrics_dict["RL_time_total"] = time_finish_episode - time_server_start #Total Server time untill now, it will be overwritten after next episode
		#Save data at the end of each episode; Overwrite ( new written metrics dicts contains old episode data + the new episode)
		#Overall Structure is metrics_dict -> episode_dict -> step_dict
		with open(config.home + '/results/RL_Metrics3.pkl','wb') as f:
					pickle.dump(metrics_dict,f)

	##Out of Episode loop
	time_server_finish= time.perf_counter()
	metrics_dict["RL_time_total"] = time_server_finish - time_server_start
	print("Finished RL_serverrun")
		

