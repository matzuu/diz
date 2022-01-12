if __name__ == "__main__":
	import time
	import torch
	import pickle
	import argparse

	import logging
	logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)

	import sys
	sys.path.append('../FedAdapt')
	from Sever import Sever
	import config
	import utils
	import PPO

	parser=argparse.ArgumentParser()
	parser.add_argument('--offload', help='FedAdapt or classic FL mode', type= utils.str2bool, default= False)
	args=parser.parse_args()

	LR = config.LR
	offload = args.offload
	first = True # First initializaiton control

	logger.info('Preparing Sever.')
	sever = Sever(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')
	##RECEIVED INCOMING CONNECTION: Start the timer for TOTAL EXECUTION TIME
	time_server_start = time.perf_counter()
	sever.initialize(config.split_layer, offload, first, LR)
	first = False

	state_dim = 2*config.G
	action_dim = config.G

	if offload:
		#Initialize trained RL agent 
		agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
		agent.policy.load_state_dict(torch.load('FL_training/PPO_FedAdapt.pth'))

	if offload:
		logger.info('FedAdapt Training')
	else:
		logger.info('Classic FL Training')

	res = {}
	res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

	for r in range(config.R):
		logger.info('====================================>')
		logger.info('==> Round {:} Start'.format(r))

		time_round_start = time.time()
		state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
		aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
		time_round_finish = time.time()

		# Recording each round training time, bandwidth and test accuracy
		time_training_round = time_round_finish - time_round_start
		res['training_time'].append(time_training_round)
		res['bandwidth_record'].append(bandwidth)

		test_acc = sever.test(r)
		res['test_acc_record'].append(test_acc)

		with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
					pickle.dump(res,f)

		logger.info('Round Finish')
		logger.info('==> Round Training Time: {:}'.format(time_training_round))

		logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
		logger.info("#### Offload" + str(offload))
		if offload:
			logger.info('Offload Train')
			split_layers = sever.adaptive_offload(agent, state)
		else:
			logger.info('NO Offload Train ')
			split_layers = config.split_layer

		if r > 49:
			LR = config.LR * 0.1

		sever.reinitialize(split_layers, offload, first, LR)
		logger.info('==> Reinitialization Finish')
	##Finished everything
	client_metrics_list = sever.receive_time_metrics()
	with open(config.home + '/results/Client_time_metrics6.pkl','wb') as f:
					pickle.dump(client_metrics_list,f)

	time_server_finish = time.perf_counter()
	logger.info(f"## Finished FedAdapt in {(time_server_finish - time_server_start)/60:0.4f} minutes, or {time_server_finish - time_server_start:0.4f} seconds")
	res['server_total__time'] = time_server_finish - time_server_start

	with open(config.home + '/results/FedAdapt_res6.pkl','wb') as f:
					pickle.dump(res,f)

