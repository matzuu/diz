from psutil import cpu_percent


def client_main():
	import torch
	import socket
	import numpy as np

	import logging
	logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)

	import sys
	sys.path.append('../FedAdapt')
	from RLEnv import RL_Client
	import config
	import utils
	import psutil
	import time

	psutil.cpu_percent() #Start monitoring Resource Usage Metrics
	#Also monitor RAM here?
	
	time_client_start = time.perf_counter()
	if config.random:
		torch.manual_seed(config.random_seed)
		np.random.seed(config.random_seed)
		logger.info('Random seed: {}'.format(config.random_seed))

	first = True
	ip_address = config.HOST2IP[socket.gethostname()]
	index = config.CLIENTS_CONFIG[ip_address]
	datalen = config.N / config.K
	split_layer = config.split_layer[index]

	logger.info('==> Preparing Data..')
	cpu_count = psutil.cpu_count()
	trainloader, classes= utils.get_local_dataloader(index, cpu_count)

	logger.info('==> Preparing RL_Client..')
	rl_client = RL_Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, config.model_name, split_layer, config.model_cfg)

	while True:
		psutil.cpu_percent() #init since last "infer"; it will wait until next msg is received; Resource Calculation is done after the infer state
		start_time = time.perf_counter()  #can be either step, or episode_init time

		command = rl_client.recv_msg(rl_client.sock, 'NEXT_COMMAND')[1]
		if command == 'RESET':
			rl_client.initialize(len(config.model_cfg[config.model_name])-1) #initialize with llen of model -1 <==> 7 - 1 = 6; NO offloading
		elif command == 'CONTINUE_COMPUTING':
			logger.info('==> Next Timestep..')
			config.split_layer = rl_client.recv_msg(rl_client.sock, 'SPLIT_LAYERS')[1]
			rl_client.reinitialize(config.split_layer[index]) #Initialize with current split_layer value, (potential) offloading
		elif command == 'RUN_FINISHED':
			time_client_finish = time.perf_counter()
			time_client_total = time_client_finish - time_client_start
			logger.info('## FINISHING_RUN: gathering metrics...')
			rl_client.send_msg_run_finished_client(time_client_total)
			return
		
		
		logger.info('==> Training Start..')
		if first:
			rl_client.infer(trainloader)
			rl_client.infer(trainloader)
			
			first = False
		else:
			rl_client.infer(trainloader)

		finish_time = time.perf_counter()  #can be either step, or episode_init time
		round_time = finish_time - start_time
		rl_client.send_RW_metrics(round_time) #Send the RW metrics
		
if __name__ == "__main__":
	client_main()
		


