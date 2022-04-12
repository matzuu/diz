import sys

# Network configration
#Server-internal = '172.20.2.75' # zerotier '172.25.185.251' # Openstack = '143.205.173.100' #'dellOptiplex43' : '143.205.122.203'
SERVER_ADDR = '143.205.122.203' #server&client-side
SERVER_PORT = 51000 #server&client-side

K = 5 # Number of devices
G = 3 # Number of groups

#'143.205.122.114' bogdan6

# Unique clients order
CLIENTS_LIST= ['143.205.122.114', '143.205.122.79' , '143.205.122.92' , '143.205.122.99' ,   '143.205.122.102']  #server&client-side
HOST2IP = {'bogdan6':CLIENTS_LIST[0], 'bogdan1':CLIENTS_LIST[1] , 'bogdan2':CLIENTS_LIST[2] , 'bogdan3':CLIENTS_LIST[3], 'bogdan4':CLIENTS_LIST[4]} #server&client-side
CLIENTS_CONFIG= {CLIENTS_LIST[0]:0 , CLIENTS_LIST[1]:1 , CLIENTS_LIST[2]:2, CLIENTS_LIST[3]:3, CLIENTS_LIST[4]:4} #server&client-side

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('FedAdapt')[0] + 'FedAdapt'
dataset_path = home +'/dataset/'+ dataset_name +'/'
print(home)
print(dataset_path)
N = 50000 # data length


# Model configration
model_cfg = { #c_out == channel_out
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in)) #De ce 2x Kernel ptr flops?
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
R = 100 # FL rounds
LR = 0.01 # Learning rate
B = 50 # Batch size


# RL training configration
max_episodes = 100         # max training episodes  #server-side-only
max_timesteps = 100       # max timesteps in one episode #server-side-only
exploration_times = 20	   # exploration times without std decay

n_latent_var = 64          # number of variables in hidden layer
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timesteps 	   
max_update_epochs = 10     ##### Stop the steps training after X values.(results in episodes with 1 step) Not sure exactly the reason for implementation.
						   ##### controls when to stop multiples steps in an episode. nr of steps before stopping: update_timestep * control_update_epoch
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
#rl_b = 100				   # Batchsize ####not used? ### Test to see how it behaves when different from B
rl_lr = 0.0003             # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {CLIENTS_LIST[0]: 5, CLIENTS_LIST[1]: 5 , CLIENTS_LIST[2]: 5, CLIENTS_LIST[3]: 5,CLIENTS_LIST[4] : 5}  # infer times for each device #server&client-side

tolerance_counts = 0 #Number of allowances for exceeding max baseline in the reward function; Default 0
tolerance_percent = 1.0 #Tolerance in regards to the percentage of baseline(maximum threshold)the current max_iteration_time is allowed to go over;  Default 1
						#I.E: max_itertime < tolerance_percent * baseline;

random = True
random_seed = 0

#Zerotier {'edge-device-1':'172.25.241.195' , 'edge-device-2':'172.25.12.59' , 'edge-device-3':'172.25.133.133' , 'edge-device-4':'172.25.135.52' , 'dellOptiplex43':'172.25.71.194'}

#More than 5 deviced; PAY ATTENTION TO UNUSED
#Openstack {'edge-device-1':'143.205.173.117' , 'edge-device-2':'143.205.173.105' , 'edge-device-3':'172.25.173.201' , 'edge-device-4':'143.205.173.119' , 'edge-device-5':'143.205.173.103', 'nishant-bogdan':'143.205.173.101'}