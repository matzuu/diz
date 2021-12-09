import sys

# Network configration
SERVER_ADDR = '172.25.185.251' #Server-internal = '172.20.2.75' # zerotier '172.25.185.251' # Openstack = '143.205.173.100' 
SERVER_PORT = 51000

K = 5 # Number of devices
G = 3 # Number of groups

# Unique clients order
HOST2IP = {'VM-Exoscale-Tiny2':'172.25.42.213' , 'VM-Exoscale-Tiny':'172.25.250.116' , 'VM-Exoscale-Small':'172.25.173.201' , 'gpu1':'143.205.173.64' , 'gpu3':'143.205.173.65'}
CLIENTS_CONFIG= {'172.25.42.213':0 , '172.25.250.116':1, '172.25.173.201':2, '143.205.173.64':3, '143.205.173.65':4}
CLIENTS_LIST= ['172.25.42.213' , '172.25.250.116' , '172.25.173.201' , '143.205.173.64' , '143.205.173.65'] 

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('FedAdapt')[0] + 'FedAdapt'
dataset_path = home +'/dataset/'+ dataset_name +'/'
print(home)
print(dataset_path)
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [6,6,6,6,6] #Initial split layers
model_len = 7


# FL training configration
R = 10 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size


# RL training configration
max_episodes = 100         # max training episodes
max_timesteps = 100        # max timesteps in one episode
exploration_times = 20	   # exploration times without std decay
n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_b = 100				   # Batchsize
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'172.25.42.213': 5 , '172.25.250.116': 5, '172.25.173.201': 5, '143.205.173.64': 5, '143.205.173.65': 5}  # infer times for each device

random = True
random_seed = 0

#Zerotier {'edge-device-1':'172.25.241.195' , 'edge-device-2':'172.25.12.59' , 'edge-device-3':'172.25.133.133' , 'edge-device-4':'172.25.135.52' , 'dellOptiplex43':'172.25.71.194'}

#More than 5 deviced; PAY ATTENTION TO UNUSED
#Openstack {'edge-device-1':'143.205.173.117' , 'edge-device-2':'143.205.173.105' , 'edge-device-3':'172.25.173.201' , 'edge-device-4':'143.205.173.119' , 'edge-device-5':'143.205.173.103', 'nishant-bogdan':'143.205.173.101'}