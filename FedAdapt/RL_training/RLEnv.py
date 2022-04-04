import psutil

from http import client
import torch
import torch.nn as nn
import torch.optim as optim

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import tqdm
import time
import numpy as np
import threading
import json
import operator


import sys
sys.path.append('../FedAdapt')
import config
import utils
from Communicator import *

if config.random:
	torch.manual_seed(config.random_seed)
	np.random.seed(config.random_seed)
	logger.info('Random seed: {}'.format(config.random_seed))

class Env(Communicator):
	def __init__(self, index, ip_address, server_port, clients_list, model_name, model_cfg, batchsize):
		super(Env, self).__init__(index, ip_address)
		self.index = index
		self.clients_list = clients_list
		self.model_name = model_name
		self.batchsize = batchsize
		self.model_cfg = model_cfg
		self.state_dim = 2*config.G 
		self.action_dim = config.G
		self.group_labels = []
		self.model_flops_list = self.get_model_flops_list(model_cfg, model_name)
		assert len(self.model_flops_list) == config.model_len
		self.client_total_time = {}

		# Server configration
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.port = server_port
		self.model_name = model_name
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #TO reuse the address
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}
		

		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock

		self.uninet = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)


	def reset(self, done, first):
		split_layers = [config.model_len-1 for i in range(config.K)] # Reset with no offloading
		config.split_layer = split_layers
		thread_number = config.K
		client_ips = config.CLIENTS_LIST
		self.tolerance_counts = config.tolerance_counts
		self.ep_cpu_wastage_overhead = dict(zip(client_ips,[0] * len(client_ips))) #initialize dict with client_ip keys and 0 values
		self.ep_ram_wastage_overhead  = dict(zip(client_ips, [0] * len(client_ips)))
		self.ep_disk_wastage_overhead  = dict(zip(client_ips, [0] * len(client_ips)))
		self.initialize(split_layers)
		msg = ['NEXT_COMMAND', 'RESET']
		self.scatter(msg)

		#Test network speed
		self.test_network(thread_number, client_ips)
		self.network_state = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')
			self.network_state[msg[1]] = msg[2]

		# Classic FL training
		if first:
			self.infer(thread_number, client_ips)
			self.infer(thread_number, client_ips)
		else:
			self.infer(thread_number, client_ips)

	

		self.offloading_state = self.get_offloading_state(split_layers, self.clients_list, self.model_cfg, self.model_name)
		self.baseline = self.infer_state # Set baseline for normalization
		if len(self.group_labels) == 0:
			self.group_model, self.cluster_centers, self.group_labels = self.group(self.baseline, self.network_state)

		logger.info('Basiline: '+ json.dumps(self.baseline))

		# Concat and norm env state
		state = self.concat_norm(self.clients_list ,self.network_state, self.infer_state, self.offloading_state)
		assert self.state_dim == len(state)

		self.receive_RW_from_clients("reset")
		return np.array(state)

	def group(self, baseline, network):
		from sklearn.cluster import KMeans
		X = []
		index = 0
		netgroup = []
		for c in self.clients_list:
			X.append([baseline[c]])

		# Clustering without network limitation
		kmeans = KMeans(n_clusters=config.G, random_state=0).fit(X)
		cluster_centers = kmeans.cluster_centers_
		labels = kmeans.predict(X)

		'''
		# Clustering with network limitation
		kmeans = KMeans(n_clusters=config.G - 1, random_state=0).fit(X)
		cluster_centers = kmeans.cluster_centers_
		labels = kmeans.predict(X)

		# We manually set Pi3_2 as seperated group for limited bandwidth
		labels[-1] = 2
		'''
		return kmeans, cluster_centers, labels


	def step(self, action, done): #TODO done is not used?
		# Expand action to each device and initialization
		action = self.expand_actions(action, self.clients_list)
		#offloadings = 1 - np.clip(np.array(action), 0, 1)
		config.split_layer = self.action_to_layer(action)
		split_layers = config.split_layer
		logger.info('Current OPs: ' + str(split_layers))
		thread_number = config.K
		client_ips = config.CLIENTS_LIST
		self.initialize(split_layers)


		self.time_start_client_processing = time.perf_counter()
		try:
			self.step_client_interstep_idle_time = self.time_start_client_processing - self.time_finish_client_processing															# 1e-5 is the tolerance for float rounding error; the substraction is small => it is 0
			# ^ Client Time idle between end of previous step and beginning of this step;
		except AttributeError: #first round; time_finish_client_processing is not initialized so it results in AttributeError; 
			self.step_client_interstep_idle_time = 0.0 #cannot exist idle time before first round;

		######################
		##Forcing No Offload##
		######################
		# config.split_layer = [6,6,6,6,6]
		# split_layers = [6,6,6,6,6]
		# self.split_layers = [6,6,6,6,6]
		#################
		msg = ['NEXT_COMMAND', 'CONTINUE_COMPUTING']
		self.scatter(msg)

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		
		#Test network speed
		self.test_network(thread_number, client_ips)
		self.network_state = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK_SPEED')
			self.network_state[msg[1]] = msg[2]

		# Offloading training and return env state
		##
		logger.info('## Infering on the server inside the step function')
		##
		
		self.infer(thread_number, client_ips)
		self.time_finish_client_processing = time.perf_counter()

		self.offloading_state = self.get_offloading_state(split_layers, self.clients_list, self.model_cfg, self.model_name)
		reward, maxtime, done = self.calculate_reward(self.infer_state)
		logger.info('Training time per iteration: '+ json.dumps(self.infer_state))
		state = self.concat_norm(self.clients_list ,self.network_state, self.infer_state, self.offloading_state)
		assert self.state_dim == len(state)

		self.receive_RW_from_clients("step")

		return np.array(state), reward, maxtime, done

	def initialize(self, split_layers):
		self.split_layers = split_layers
		self.nets = {}
		self.optimizers = {}
		self.step_client_offloading_idle_time = {}
		self.step_client_interstep_idle_time = 0.0 #Will be modified if it's not the first step and Val will be modified;
		self.cpu_wastage_step = {}
		self.ram_wastage_step = {}
		self.disk_wastage_step= {}

		for i in range(len(split_layers)):
			client_ip = config.CLIENTS_LIST[i]
			self.step_client_offloading_idle_time[client_ip] = [] #Iteration Idle time
			if split_layers[i] < config.model_len -1: # Only offloading client need initialized in server
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, self.model_cfg)
				self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=config.LR,
					  momentum=0.9)

		self.criterion = nn.CrossEntropyLoss()

	def test_network(self, thread_number, client_ips):
		self.net_threads = {}
		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
			self.net_threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.net_threads[client_ips[i]].join()

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK_SPEED')
		msg = ['MSG_TEST_NETWORK_SPEED', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def infer(self, thread_number, client_ips): #Server infer
		self.threads = {}
		for i in range(len(client_ips)):
			if self.split_layers[i] == config.model_len -1:
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_infer_no_offloading, args=(client_ips[i],))
				logger.debug(str(client_ips[i]) + ' no offloading infer start')
				self.threads[client_ips[i]].start()
			else:
				logger.debug(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_infer_offloading, args=(client_ips[i],))
				logger.debug(str(client_ips[i]) + ' offloading infer start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

		self.infer_state = {}
		self.total_iterations_time = {}
		self.iteration_metrics = {}
		time_start_idle_server = time.perf_counter()
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_INFER_SPEED')
			self.infer_state[msg[1]] = msg[2]
			

			self.total_iterations_time[msg[1]] = msg[3]  #msg[1] = 'ip.addres'
			self.iteration_metrics[msg[1]] = msg[4]
		time_finish_idle_server = time.perf_counter()
		#Time it takes to wait for the clients to send their finish tasks msg. (MSG_INFER_SPEED)
		self.server_idle_time = time_finish_idle_server - time_start_idle_server 
		

	def _thread_infer_no_offloading(self, client_ip):
		pass

	def _thread_infer_offloading(self, client_ip):
		for i in range(config.iteration[client_ip]):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
			#After the clients sends this message, he stays idle:
			time_start_iteration_idleTime_client = time.perf_counter()

			smashed_layers = msg[1]
			labels = msg[2]

			self.optimizers[client_ip].zero_grad()
			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			outputs = self.nets[client_ip](inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()

			self.optimizers[client_ip].step()

			# Send gradients to client
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)
			# After the client receives this message, he no longer stays idle:
			time_finish_iteration_idleTime_client = time.perf_counter()
			self.step_client_offloading_idle_time[client_ip].append(time_finish_iteration_idleTime_client - time_start_iteration_idleTime_client)


	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

	def get_offloading_state(self, split_layer, clients_list, model_cfg, model_name):
		offloading_state = {}
		offload = 0

		assert len(split_layer) == len(clients_list)
		for i in range(len(clients_list)):
			for l in range(len(model_cfg[model_name])):
				if l <= split_layer[i]:
					offload += model_cfg[model_name][l][5]

			offloading_state[clients_list[i]] = offload / config.total_flops
			offload = 0

		return offloading_state

	def get_model_flops_list(self, model_cfg, model_name):
		model_state_flops = []
		cumulated_flops = 0

		for layer in model_cfg[model_name]:
			cumulated_flops += layer[5]
			model_state_flops.append(cumulated_flops)

		model_flops_list = np.array(model_state_flops)
		model_flops_list = model_flops_list / cumulated_flops

		return model_flops_list

	def concat_norm(self, clients_list ,network_state, infer_state, offloading_state):
	
		network_state_order = []
		infer_state_order = []
		offloading_state_order =[]
		for c in clients_list:
			network_state_order.append(network_state[c])
			infer_state_order.append(infer_state[c])
			offloading_state_order.append(offloading_state[c])

		group_max_index = [0 for i in range(config.G)]
		group_max_value = [0 for i in range(config.G)]
		for i in range(len(clients_list)):
			label = self.group_labels[i]
			if infer_state_order[i] >= group_max_value[label]:
				group_max_value[label] = infer_state_order[i]
				group_max_index[label] = i

		infer_state_order = np.array(infer_state_order)[np.array(group_max_index)]
		offloading_state_order = np.array(offloading_state_order)[np.array(group_max_index)]
		network_state_order = np.array(network_state_order)[np.array(group_max_index)]
		state = np.append(infer_state_order, offloading_state_order)
		return state

	def calculate_reward(self, infer_state): #TODO, change how the reward function is calculated; instead of iter time, max client step time?
		rewards = {}
		reward = 0
		done = False

		max_basetime = max(self.baseline.items(), key=operator.itemgetter(1))[1]
		max_infertime = max(infer_state.items(), key=operator.itemgetter(1))[1]
		max_infertime_index = max(infer_state.items(), key=operator.itemgetter(1))[0]
			   
		if max_infertime >= config.tolerance_percent * max_basetime: #Default tolerance_percent == 1.0
			self.tolerance_counts -= 1
			if self.tolerance_counts < 0:
				done = True
				#reward += - 1
		else:
			done = False

		for k in infer_state:
			if infer_state[k] < self.baseline[k]:
				r = (self.baseline[k] - infer_state[k]) / self.baseline[k]
				reward += r
			else:
				r = (infer_state[k] - self.baseline[k]) / infer_state[k]
				reward -= r	

		return reward, max_infertime, done

	def expand_actions(self, actions, clients_list):
		full_actions = []

		for i in range(len(clients_list)):
			full_actions.append(actions[self.group_labels[i]])

		return full_actions

	def action_to_layer(self, action):
		split_layer = []
		for v in action:
			idx = np.where(np.abs(self.model_flops_list - v) == np.abs(self.model_flops_list - v).min()) 
			idx = idx[0][-1]
			if idx >= 5: # all FC layers combine to one option
				idx = 6
			split_layer.append(idx)
		return split_layer

	def run_finished_metrics_client_handling(self):
		msg = ['NEXT_COMMAND','RUN_FINISHED']
		self.scatter(msg)
		## Wait for return message with metrics

		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'RUN_FINISHED_METRICS')
			self.client_total_time[msg[1]] = msg[2]

		#Send close signal to other devices
		return

	def close_connections(self):
		for s in self.client_socks:
			self.client_socks[s].shutdown(socket.SHUT_RDWR)
			self.client_socks[s].close()
		return

	def calculate_resource_wastage_server(self,time_total):
		cpu_count,cpu_usage_percent,ram_usage,disk_usage = get_resource_utilisation()
		cpu_wastage,ram_wastage,disk_wastage = calculate_resource_wastage(time_total,cpu_count,cpu_usage_percent,ram_usage,disk_usage)
		return cpu_wastage,ram_wastage,disk_wastage

	def receive_RW_from_clients(self,context):
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'RESOURCE_WASTAGE')
			#msg = ['RESOURCE_WASTAGE', self.ip,cpu_RW,ram_RW,disk_RW ] #msg received

			if context == 'reset': 
				self.ep_cpu_wastage_overhead[msg[1]] += msg[2]
				self.ep_ram_wastage_overhead[msg[1]] += msg[3]
				self.ep_disk_wastage_overhead[msg[1]] += msg[4]
			elif context == 'step':
				self.cpu_wastage_step[msg[1]] = msg[2]
				self.ram_wastage_step[msg[1]] = msg[3]
				self.disk_wastage_step[msg[1]] = msg[4]


		

class RL_Client(Communicator):
	def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer, model_cfg):
		super(RL_Client, self).__init__(index, ip_address)
		self.ip_address = ip_address
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.model_cfg = model_cfg
		self.uninet = utils.get_model('Unit', self.model_name, 0, self.device, self.model_cfg)

		logger.info('==> Connecting to Server..')
		self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer):
		self.split_layer = split_layer
		self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, self.model_cfg)
		self.optimizer = optim.SGD(self.net.parameters(), lr=config.LR,
					  momentum=0.9)
		self.criterion = nn.CrossEntropyLoss()

		# First test network speed
		network_time_start = time.time()
		msg = ['MSG_TEST_NETWORK_SPEED', self.uninet.cpu().state_dict()]
		self.send_msg(self.sock, msg)
		msg = self.recv_msg(self.sock,'MSG_TEST_NETWORK_SPEED')[1]
		network_time_end = time.time()
		network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start) # format is Mbit/s 

		msg = ['MSG_TEST_NETWORK_SPEED', self.ip, network_speed]
		self.send_msg(self.sock, msg)

	def infer(self, trainloader):  #Client Infer
		self.net.to(self.device)
		self.net.train()
		metrics_current_infer = {}
		s_time_infer = time.perf_counter()
		
		if self.split_layer == len(config.model_cfg[self.model_name]) -1: # No offloading  #model_cfg is a dict with 1 element with the key: "VGG5" and Value: Array of 7 elements; if split_layer is 6 then len(value) - 1 == 6;
			for batch_idx, (inputs, targets, indexes) in enumerate(tqdm.tqdm(trainloader)): #Enumerate makes multi-process dataloader 
				s_time_batch_infer = time.perf_counter()	

				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.net(inputs)
				##NonOffloading
				loss = self.criterion(outputs, targets)
				loss.backward()
				##NonOffloading
				self.optimizer.step()

				######Metrics Gathering
				f_time_batch_infer = time.perf_counter()
				iteration_time = f_time_batch_infer - s_time_batch_infer
				metrics_current_infer['Batch_'+str(batch_idx)] = (iteration_time,indexes)
				######Done Metrics Gathering

				if batch_idx >= config.iteration[self.ip_address]-1:
					break
		else: # Offloading training
			for batch_idx, (inputs, targets, indexes) in enumerate(tqdm.tqdm(trainloader)):
				s_time_batch_infer = time.perf_counter()
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.net(inputs)
				##Offloading  ##Send some tasks to the server #TODO check serverside
				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
				self.send_msg(self.sock, msg)
					# Wait receiving server gradients 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'
				gradients = self.recv_msg(self.sock)[1].to(self.device)
				outputs.backward(gradients)
				##Offfloading
				self.optimizer.step()

				######Metrics Gathering
				f_time_batch_infer = time.perf_counter()
				iteration_time = f_time_batch_infer - s_time_batch_infer
				metrics_current_infer['Batch_'+str(batch_idx)] = (iteration_time,indexes)
				######Done Metrics Gathering

				if batch_idx >= config.iteration[self.ip_address]-1:
					break

		e_time_infer = time.perf_counter()
		logger.info('Training time: ' + str(e_time_infer - s_time_infer))

		infer_speed = (e_time_infer - s_time_infer) / config.iteration[self.ip_address]
		infer_total_time = e_time_infer - s_time_infer
		

		msg = ['MSG_INFER_SPEED', self.ip, infer_speed,infer_total_time, metrics_current_infer]
		self.send_msg(self.sock, msg)

		

	def reinitialize(self, split_layers):
		self.initialize(split_layers)

	def send_msg_run_finished_client(self, time_client_total):
		msg = ['RUN_FINISHED_METRICS', self.ip, time_client_total] 
		self.send_msg(self.sock, msg)
		return

	def send_RW_metrics(self, time_client_total):
		cpu_RW,ram_RW,disk_RW = self.calculate_resource_wastage_client(time_client_total)
		msg = ['RESOURCE_WASTAGE', self.ip,cpu_RW,ram_RW,disk_RW ]
		self.send_msg(self.sock, msg)
		return

	def calculate_resource_wastage_client(self,time_client_total):
		cpu_count,cpu_usage_percent,ram_usage,disk_usage = get_resource_utilisation()
		return calculate_resource_wastage(time_client_total,cpu_count,cpu_usage_percent,ram_usage,disk_usage)



def get_resource_utilisation():

	cpu_usage_percent = psutil.cpu_percent() #returns percent of utilization since last call; if no other call returns 0.0 ==> must call this at least once before
	cpu_count = psutil.cpu_count()
	ram_usage = psutil.virtual_memory()
	disk_usage = psutil.disk_usage('/')	

	return cpu_usage_percent,cpu_count,ram_usage,disk_usage

def calculate_resource_wastage(time_client_total,cpu_count,cpu_usage_percent,ram_usage,disk_usage):
	#CPU resource wastage == %idle_time * nr_cpus * time(s)
	cpu_RW = ((100.0-cpu_usage_percent)/100) * cpu_count * time_client_total
	# RAM wastage == GB_count of available/free ram *time(s)
	ram_available = ram_usage.available / (1024*1024*1024) # Available_bytes/1GB 
	ram_RW = ram_available * time_client_total
	# DISK wastage == 5gb ; doesn't make sense for time?
	disk_available = disk_usage.free / (1024*1024*1024) # Available_bytes/1GB
	disk_RW = (disk_available/5) * time_client_total
	return cpu_RW,ram_RW,disk_RW	