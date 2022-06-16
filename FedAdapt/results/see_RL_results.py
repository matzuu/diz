import pickle #
import matplotlib.pyplot as plt
import numpy as np
import operator
import inspect
import sys
sys.path.append('../FedAdapt')
import config
import processing.entropy as entro
##IMPORTS##
#########################################




def display_split_layer_by_episode(RL_res1):
    split_layer_matrix = []

    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot
        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            split_layer_matrix.append(step['split_layer'])

            

    x = range(1,len(split_layer_matrix)+1)
    y1 = np.array(split_layer_matrix)[:,0] #Values of 1st client
    y2 = np.array(split_layer_matrix)[:,1]#Values of 2nd client
    y3 = np.array(split_layer_matrix)[:,2]
    y4 = np.array(split_layer_matrix)[:,3]
    y5 = np.array(split_layer_matrix)[:,4]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(x, y1 color='tab:red')
    #ax.plot(x, y2, color='tab:orange')
    #ax.plot(x, y3, color='tab:green')
    #ax.plot(x, y4, color='tab:blue')
    ax.plot(x, y5, color='tab:purple')

    ax.set(xlabel='step nr', ylabel=' split_layer ',
        title='split_layer progress over time')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(1, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.8)
    
    ax.legend(loc="upper right")

    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("Average of 1st device split_layer = " + str(np.average(y1)) )
    print("Std of 1st device split_layer = " + str(np.std(y1)) )
    print("Average of 2nd device split_layer = " + str(np.average(y2)) )
    print("Std of 2nd device split_layer = " + str(np.std(y2)) )
    #print("Average of 3rd device split_layer = " +str( np.average(y3)) )
    #print("Std of 3rd device split_layer = " + np.std(y3) )
    #print("Average of 3th device split_layer = " + np.average(y4) )
    #print("Std of 4th device split_layer = " + np.std(y4) )
    #print("Average of 5th device split_layer = " + np.average(y5) )
    #print("Std of 5th device split_layer = " + np.std(y5) )

    #METRICS DICT -> EPISODES_DICT -> STEPS DICT -> small metrics
    print("DONE_DISPLAY")

def display_steps_and_relativeTime_per_episode(RL_res1):
    episode_nrSteps_list = []
    episode_avg_step_time = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]

    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)
        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        episode_nrSteps_list.append(len(episode)-1)
        episode_avg_step_time.append(episode["episode_time_total"]/len(episode)-1)

        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

    ##########################################################
    x = range(1,len(episode_nrSteps_list)+1)
    y1 = np.array(episode_nrSteps_list) 
    y2 = np.array(episode_avg_step_time)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:red',label="Steps Count")
    ax.plot(x, y2, color='tab:blue',label="Avg Time per Step") #Can also consider Total Time/ #Steps

    ax.set(xlabel='Episode Nr', ylabel=' counts/time ',
        title='Nr of steps vs avg step time, per episode')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(1, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.8)
    
    ax.legend(loc="upper right")

    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()


    print("DONE_DISPLAY")

def display_eachStep_rew_maxIterTime_stepTime(RL_res1):
    step_reward_list = []
    step_maxIter_list = []
    step_totalTimeServer_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_reward_list.append(step["rewards"])
            step_maxIter_list.append(step["maxtime_iteration"])
            step_totalTimeServer_list.append(step["server_step_time_total"])

    ##########################################################
    x = range(1,len(step_reward_list)+1)
    y1 = np.array(step_reward_list) 
    y2 = np.array(step_maxIter_list)
    y3 = np.array(step_totalTimeServer_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:green',label="Rewards")
    ax.plot(x, y2, color='tab:red',label="Max Iteration Time")
    ax.plot(x, y3, color='tab:blue',label="Server Step Time")
    ax.set(xlabel='Step Nr', ylabel=' Reward Val / time(seconds) ',
        title='Reward vs Max Iteration Time vs Server Step Time')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()


    print("DONE_DISPLAY")

def display_maxIterTime(RL_res1):
    step_maxIter_list = []
    step_maxBaseline_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_maxIter_list.append(step["maxtime_iteration"])
            current_maxBaseline = max(step["client_baseline"].items(), key=operator.itemgetter(1))[1]
            step_maxBaseline_list.append(current_maxBaseline)

    ##########################################################
    x = range(1,len(step_maxIter_list)+1)
    y1 = np.array(step_maxBaseline_list)
    y2 = np.array(step_maxIter_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:red',label="Max Baseline Iteration")
    ax.plot(x, y2, color='tab:blue',label="Max Iteration Time")
    ax.set(xlabel='Step Nr', ylabel=' Time (seconds)',
        title='Max Iteration Time')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 1.1, 0.25)
    minor_y_ticks = np.arange(0, 1.1, 0.1)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_server_and_client_steptime(RL_res1):
    step_serverTotalTime_list = []
    step_clientTotalTime_dict = {'143.205.122.114':[],'143.205.122.79':[] , '143.205.122.92':[] , '143.205.122.99':[] , '143.205.122.102':[]}
    
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverTotalTime_list.append(step["server_step_time_total"])
            for key in step["client_step_time_total"].keys():
                step_clientTotalTime_dict[key].append(step["client_step_time_total"][key])
                
           
    ##########################################################
    x = range(1,len(step_serverTotalTime_list)+1)
    y1 = np.array(step_serverTotalTime_list) 
    key_list = list(step_clientTotalTime_dict.keys())
    y2 = np.array(step_clientTotalTime_dict[key_list[0]])
    y3 = np.array(step_clientTotalTime_dict[key_list[1]])
    y4 = np.array(step_clientTotalTime_dict[key_list[2]])
    y5 = np.array(step_clientTotalTime_dict[key_list[3]])
    y6 = np.array(step_clientTotalTime_dict[key_list[4]])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Server Time")
    ax.plot(x, y2, color='tab:green',label="Client 6")
    ax.plot(x, y3, color='tab:brown',label="Client 1")
    ax.plot(x, y4, color='tab:orange',label="Client 2")
    ax.plot(x, y5, color='tab:red',label="Client 3")
    ax.plot(x, y6, color='tab:purple',label="Client 4")
    ax.set(xlabel='Step Nr', ylabel=' time(seconds) ',
        title='Server and clients step times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(1, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_server_and_client_maxSteptime(RL_res1):
    step_serverTotalTime_list = []
    step_clientMaxTime = []
    step_clientSumTime = []
    
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverTotalTime_list.append(step["server_step_time_total"])
            max_val = max(step["client_step_time_total"].values())
            step_clientMaxTime.append(max_val)
            sum_vals = sum(step["client_step_time_total"].values())
            step_clientSumTime.append(sum_vals/5)

                
           
    ##########################################################
    x = range(1,len(step_serverTotalTime_list)+1)
    y1 = np.array(step_serverTotalTime_list)
    y2 = np.array(step_clientMaxTime)
    y3 = np.array(step_clientSumTime)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Server Time")
    ax.plot(x, y2, color='tab:green',label="Max Client Time")
    ax.plot(x, y3, color='tab:brown',label="Sum Clients Time")
    ax.set(xlabel='Step Nr', ylabel=' time(seconds) ',
        title='Server and Max/Sum of clients step times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()
    

    print("DONE_DISPLAY")

def display_server_and_client_idle_time(RL_res1):
    step_serverIdleTime_list = []
    step_clientInterStepIdle_list = []
    step_clientOffloadIdle_dict = {'143.205.122.114':[],'143.205.122.79':[] , '143.205.122.92':[] , '143.205.122.99':[] , '143.205.122.102':[]}
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverIdleTime_list.append(step["server_idle_time"])
            step_clientInterStepIdle_list.append(step["client_interstep_idle_time"])
            for key in step["client_offloading_idle_time"].keys():
                if len(step["client_offloading_idle_time"][key]) > 0: #i.e the list is not empty (because of offloading)
                    step_clientOffloadIdle_dict[key].append(sum(step["client_offloading_idle_time"][key])) #get Sum of idle iterrations time in a step, for a client
                else:
                    step_clientOffloadIdle_dict[key].append(0) #i.e the list is empty, because there is no iddle time in no offloading
                
           
    ##########################################################
    x = range(1,len(step_clientInterStepIdle_list)+1)
    y1 = np.array(step_clientInterStepIdle_list) 
    key_list = list(step_clientOffloadIdle_dict.keys())
    y2 = np.array(step_clientOffloadIdle_dict[key_list[0]])
    y3 = np.array(step_clientOffloadIdle_dict[key_list[1]])
    y4 = np.array(step_clientOffloadIdle_dict[key_list[2]])
    y5 = np.array(step_clientOffloadIdle_dict[key_list[3]])
    y6 = np.array(step_clientOffloadIdle_dict[key_list[4]])
    y7 = np.array(step_serverIdleTime_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:cyan',label="Interstep Idle Time")
    ax.plot(x, y2, color='tab:green',label="Client 6")
    ax.plot(x, y3, color='tab:brown',label="Client 1")
    ax.plot(x, y4, color='tab:orange',label="Client 2")
    ax.plot(x, y5, color='tab:red',label="Client 3")
    ax.plot(x, y6, color='tab:purple',label="Client 4")
    ax.plot(x, y7, color='tab:blue',label="Server Idle Time")
    ax.set(xlabel='Step Nr', ylabel=' time(seconds) ',
        title='Server and clients idle times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_server_and_client_maxNmin_idle_time(RL_res1):
    step_serverIdleTime_list = []
    step_clientInterStepIdle_list = []
    step_clientMaxIdle_list = []
    step_clientMinIdle_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverIdleTime_list.append(step["server_idle_time"])
            step_clientInterStepIdle_list.append(step["client_interstep_idle_time"])
            aux_list = step["client_offloading_idle_time"].values()
            step_clientMaxIdle_list.append(sum(max(aux_list, key = sum)))
            step_clientMinIdle_list.append(sum(min(aux_list, key = sum)))
                
           
    ##########################################################
    x = range(1,len(step_clientInterStepIdle_list)+1)
    y1 = np.array(step_clientInterStepIdle_list) 
    y2 = np.array(step_clientMaxIdle_list)
    y3 = np.array(step_clientMinIdle_list)
    y4 = np.array(step_serverIdleTime_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:orange',label="Interstep Idle Time")
    ax.plot(x, y2, color='tab:red',label="Max Sum Idle")
    ax.plot(x, y3, color='tab:green',label="Min Sum Idle")
    ax.plot(x, y4, color='tab:blue',label="Server Idle Time")
    ax.set(xlabel='Step Nr \n Sums are sums of all iterations of a client \n Min & Max are smallest/largest of those 5 sums, per step', ylabel=' time(seconds) ',
        title='Server and clients idle times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_server_client_stepNidle_time(RL_res1):
    step_serverStepTime_list = []
    step_clientStepMinTime_list = []
    step_clientStepMaxTime_list = []
    step_clientMaxIdle_list = []
    step_clientMinIdle_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverStepTime_list.append(step["server_step_time_total"])
            step_clientStepMinTime_list.append(min(step["client_step_time_total"].values()))
            step_clientStepMaxTime_list.append(max(step["client_step_time_total"].values()))
            aux_list = step["client_offloading_idle_time"].values()
            step_clientMaxIdle_list.append(sum(max(aux_list, key = sum)))
            step_clientMinIdle_list.append(sum(min(aux_list, key = sum)))
                
           
    ##########################################################
    x = range(1,len(step_serverStepTime_list)+1)
    y1 = np.array(step_clientStepMinTime_list) 
    y2 = np.array(step_clientStepMaxTime_list) 
    y3 = np.array(step_clientMaxIdle_list)
    y4 = np.array(step_clientMinIdle_list)
    y5 = np.array(step_serverStepTime_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:olive',label="Client Step Min Time")
    ax.plot(x, y2, color='tab:brown',label="Client Step Max Time")
    ax.plot(x, y3, color='tab:red',label="Max Sum Idle")
    ax.plot(x, y4, color='tab:green',label="Min Sum Idle")
    ax.plot(x, y5, color='tab:blue',label="Server Step Time")
    ax.set(xlabel='Step Nr ', ylabel=' time(seconds) ',
        title='Server and clients idle times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_1st_client_splitLayers_and_idle_time(RL_res1):
    step_serverStepTime_list = []
    step_client_layer_list = []
    step_client_stepTime_list = []
    step_client_idleTime_list = []

    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverStepTime_list.append(step['server_step_time_total'])
            step_client_layer_list.append(step['split_layer'][0])
            step_client_stepTime_list.append(step['client_step_time_total'][config.CLIENTS_LIST[0]])
            step_client_idleTime_list.append(sum(step['client_offloading_idle_time'][config.CLIENTS_LIST[0]])) #sum of all iterations idle time for 1st client
                
           
    ##########################################################
    x = range(1,len(step_serverStepTime_list)+1)
    y1 = np.array(step_serverStepTime_list) 
    y2 = np.array(step_client_layer_list) 
    y3 = np.array(step_client_stepTime_list)
    y4 = np.array(step_client_idleTime_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Server Step Time")
    ax.plot(x, y2, color='tab:olive',label="Client Layer Nr")
    ax.plot(x, y3, color='tab:red',label="Client Step Time")
    ax.plot(x, y4, color='tab:green',label="Client Idle Time")
    
    ax.set(xlabel='Step Nr ', ylabel=' time(seconds) ',
        title='Server and 1st client idle times')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()

    print("DONE_DISPLAY")

def display_add_client_interstep_time(RL_res1):
    step_serverStepTime_list = []
    step_clientStepMinSumTime_list = []
    step_clientStepMaxSumTime_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_serverStepTime_list.append(step["server_step_time_total"])
            step_clientStepMinSumTime_list.append(min(step["client_step_time_total"].values()) + step['client_interstep_idle_time'])
            step_clientStepMaxSumTime_list.append(max(step["client_step_time_total"].values()) + step['client_interstep_idle_time'])
                
     ##########################################################
    x = range(1,len(step_serverStepTime_list)+1)
    y1 = np.array(step_clientStepMaxSumTime_list) 
    y2 = np.array(step_clientStepMinSumTime_list) 
    y3 = np.array(step_serverStepTime_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:red',label="Client Interstep_idle + Max Step Time")
    ax.plot(x, y2, color='tab:green',label="Client Interstep_idle + Min Step Time")
    ax.plot(x, y3, color='tab:blue',label="Server Step Time")
    ax.set(xlabel='Step Nr ', ylabel=' time(seconds) ',
        title="Server's step and Clients(Step+Interstep) times")

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 10, 1)
    minor_y_ticks = np.arange(0, 10, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    current_function_name = inspect.stack()[0][3]
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")

def display_entropyOnAvg_batches(RL_res1):
    current_function_name = inspect.stack()[0][3]

    batch_time_list = []
    batch_imgs_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    count = 1
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(count) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            first_client_iteration = step['client_iteration_metrics'][config.CLIENTS_LIST[0]]
            for batch in first_client_iteration.values():
                
                batch_time_list.append(batch[0])
                batch_imgs_list.append(batch[1])
                count+=1
    
    entro_of_batches = entro.get_avg_entropy_of_img_batches(batch_imgs_list)
                    

    covariance = np.cov(batch_time_list,entro_of_batches)
    print("COVARIANCE "+current_function_name+":")
    print(covariance)
    ###########################################################
    x = range(1,len(batch_time_list)+1)
    y1 = np.array(batch_time_list) 
    y2 = np.array(entro_of_batches) 
    #y3 = np.array(np.divide(entro_of_batches,batch_time_list)) #checking to see how they differ

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Batch/Iteration Time List")
    ax.plot(x, y2, color='tab:red',label="Entropy")
    #ax.plot(x, y3, color='tab:red',label="Entropy/Time")
    ax.set(xlabel='Batch/Iteration Nr ', ylabel=' Entropy/time(seconds) ',
        title="Server's step and Clients(Step+Interstep) times")

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, count, 25)
    minor_x_ticks = np.arange(0, count, 5)
    major_y_ticks = np.arange(0, 8, 1)
    minor_y_ticks = np.arange(0, 8, 0.5)
    episode_stepsNr_indexes.append(count)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")

    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")

def display_entropyAggregated_batches(RL_res1):
    current_function_name = inspect.stack()[0][3]

    batch_time_list = []
    batch_imgs_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    count = 1
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(count) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            first_client_iteration = step['client_iteration_metrics'][config.CLIENTS_LIST[0]]
            for batch in first_client_iteration.values():
                
                batch_time_list.append(batch[0])
                batch_imgs_list.append(batch[1])
                count+=1
    
    entro_of_batches = entro.get_entropy_of_aggregated_img_batches(batch_imgs_list)
                    

    covariance = np.cov(batch_time_list,entro_of_batches)
    print("COVARIANCE "+current_function_name+":")
    print(covariance)
    ###########################################################
    x = range(1,len(batch_time_list)+1)
    y1 = np.array(batch_time_list) 
    y2 = np.array(entro_of_batches) 
    #y3 = np.array(np.divide(entro_of_batches,batch_time_list)) #checking to see how they differ

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Batch/Iteration Time List")
    ax.plot(x, y2, color='tab:red',label="Entropy")
    #ax.plot(x, y3, color='tab:red',label="Entropy/Time")
    ax.set(xlabel='Batch/Iteration Nr ', ylabel=' Entropy/time(seconds) ',
        title="Server's step and Clients(Step+Interstep) times")

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, count, 25)
    minor_x_ticks = np.arange(0, count, 5)
    major_y_ticks = np.arange(0, 8, 1)
    minor_y_ticks = np.arange(0, 8, 0.5)
    episode_stepsNr_indexes.append(count)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")

    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")

def display_entropy_batches_avg_vs_aggregated(RL_res1):
    current_function_name = inspect.stack()[0][3]
    batch_time_list = []
    batch_imgs_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    count = 1
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(count) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            first_client_iteration = step['client_iteration_metrics'][config.CLIENTS_LIST[0]]
            for batch in first_client_iteration.values():
                
                batch_time_list.append(batch[0])
                batch_imgs_list.append(batch[1])
                count+=1
    
    entro_of_batches_avg = entro.get_avg_entropy_of_img_batches(batch_imgs_list)
    entro_of_batches_agg = entro.get_entropy_of_aggregated_img_batches(batch_imgs_list)

    covariance = np.cov(entro_of_batches_avg,entro_of_batches_agg)
    print("COVARIANCE "+current_function_name+":")
    print(covariance)
    ###########################################################
    x = range(1,len(batch_time_list)+1)
    y1 = np.array(entro_of_batches_avg) 
    y2 = np.array(entro_of_batches_agg) 
    #y3 = np.array(np.divide(entro_of_batches,batch_time_list)) #checking to see how they differ

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="average")
    ax.plot(x, y2, color='tab:red',label="aggregated")
    #ax.plot(x, y3, color='tab:red',label="Entropy/Time")
    ax.set(xlabel='Batch/Iteration Nr ', ylabel=' Entropy ',
        title="Entropy of batches")

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, count, 25)
    minor_x_ticks = np.arange(0, count, 5)
    major_y_ticks = np.arange(0, 8, 1)
    minor_y_ticks = np.arange(0, 8, 0.5)
    episode_stepsNr_indexes.append(count)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      

def display_surprise_vs_stepTime_ALL_splitLayer(RL_res1):
    current_function_name = inspect.stack()[0][3]

    splitLayer_list = []
    client_max_step_time_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            splitLayer_list.append(step['split_layer'])
            client_max_step_time_list.append(max(step['client_step_time_total'].values()))
            
    
    total_entropy = entro.get_splitLayer_entropy(splitLayer_list)
    surprise_list = entro.get_all_splitLayer_surprise(splitLayer_list)
    covariance = np.cov(surprise_list,client_max_step_time_list)
    correlation = np.corrcoef(surprise_list,client_max_step_time_list)
    print("COVARIANCE "+current_function_name+":")
    print(covariance)
    print("CORRELATION:")
    print(correlation)
    print("ENTROPY:")
    print(total_entropy)
    ###########################################################
    x = range(1,len(surprise_list)+1)
    y1 = np.array(client_max_step_time_list) 
    y2 = np.array(surprise_list) 
    #y3 = np.array(np.divide(entro_of_batches,batch_time_list)) #checking to see how they differ

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Client Max Step Time")
    ax.plot(x, y2, color='tab:red',label="Surprise SplitLayer")
    #ax.plot(x, y3, color='tab:red',label="Entropy/Time")
    ax.set(xlabel='Step Nr', ylabel=' Surprise/time(seconds) ',
        title="SplitLayer Surprise vs ClientMax Step Time")

    # Major ticks every 20, minor ticks every 5
    #major_x_ticks = np.arange(0, count, 25)
    minor_x_ticks = np.arange(min(x), nrSteps, 5)
    major_y_ticks = np.arange(0, max(max(y1),max(y2))*1.05, 1)
    minor_y_ticks = np.arange(0, max(max(y1),max(y2))*1.05, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")

def display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res1,clientIdx):
    current_function_name = inspect.stack()[0][3]

    client_split_layer_list = []
    client_step_time_list = []
    client_idle_time_list = []
    nrSteps = 1
    episode_stepsNr_indexes = [nrSteps]
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]

            client_split_layer_list.append(step['split_layer'][clientIdx])
            client_step_time_list.append(step['client_step_time_total'][config.CLIENTS_LIST[clientIdx]])
            #client_idle_time_list.append()
        

    total_entropy = entro.get_splitLayer_entropy(client_split_layer_list)
    surprise_list = entro.get_all_splitLayer_surprise(client_split_layer_list)
    covariance = np.cov(surprise_list,client_step_time_list)
    correlation = np.corrcoef(surprise_list,client_step_time_list)
    print("COVARIANCE "+current_function_name+":")
    print(covariance)
    print("CORRELATION:")
    print(correlation)
    print("ENTROPY:")
    print(total_entropy)
    ###########################################################
    x = range(1,len(surprise_list)+1)
    y1 = np.array(client_step_time_list) 
    y2 = np.array(surprise_list) 
    #y3 = np.array(np.divide(entro_of_batches,batch_time_list)) #checking to see how they differ

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1, color='tab:blue',label="Client Step Time")
    ax.plot(x, y2, color='tab:red',label="Surprise SplitLayer")
    #ax.plot(x, y3, color='tab:red',label="Entropy/Time")
    ax.set(xlabel='Step Nr', ylabel=' Surprise/time(seconds) ',
        title="SplitLayer Surprise vs Client Step Time; CLIENT "+str(clientIdx))

    # Major ticks every 20, minor ticks every 5
    #major_x_ticks = np.arange(0, count, 25)
    minor_x_ticks = np.arange(min(x), nrSteps, 5)
    major_y_ticks = np.arange(min(x), max(max(y1),max(y2))*1.05, 1)
    minor_y_ticks = np.arange((min(x)), max(max(y1),max(y2))*1.05, 0.5)
    episode_stepsNr_indexes.append(nrSteps)
    episode_x_ticks = episode_stepsNr_indexes
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(episode_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    #ax.set_yticks(major_y_ticks)
    #ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.legend(loc="upper right")
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")

def display_boxplot_tolerance_vs_steptime_SingleClient(RL_res1,RL_res2,RL_res3,clientIdx):
    current_function_name = inspect.stack()[0][3]

    client_split_layer_matrix = [[],[],[]] #used for surprise calculation
    client_step_time_matrix = [[],[],[]]
    client_surprise_matrix = [[],[],[]]

    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)
        
        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        
        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]

            client_split_layer_matrix[0].append(step['split_layer'][clientIdx])
            client_step_time_matrix[0].append(step['client_step_time_total'][config.CLIENTS_LIST[clientIdx]])
            #client_idle_time_list.append()
    
        episode = RL_res2["episode_"+str(episode_index)] #Get Episode value
        
        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]

            client_split_layer_matrix[1].append(step['split_layer'][clientIdx])
            client_step_time_matrix[1].append(step['client_step_time_total'][config.CLIENTS_LIST[clientIdx]])
            #client_idle_time_list.append()
        episode = RL_res3["episode_"+str(episode_index)] #Get Episode value
        
        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]

            client_split_layer_matrix[2].append(step['split_layer'][clientIdx])
            client_step_time_matrix[2].append(step['client_step_time_total'][config.CLIENTS_LIST[clientIdx]])
            #client_idle_time_list.append()
        
    print("IN_FUNCTION: "+current_function_name)
    for i in range(3):  

        total_entropy = entro.get_splitLayer_entropy(client_split_layer_matrix[i])
        surprise_list = entro.get_all_splitLayer_surprise(client_split_layer_matrix[i])
        client_surprise_matrix[i] = surprise_list
        covariance = np.cov(surprise_list,client_step_time_matrix[i])
        correlation = np.corrcoef(surprise_list,client_step_time_matrix[i])
        print("#######################\n## DATA "+str(i)+" ###")
        print("COVARIANCE:")
        print(covariance)
        print("CORRELATION:")
        print(correlation)
        print("ENTROPY:")
        print(total_entropy)
        print("#######################")
    ###########################################################
    data = client_step_time_matrix
    
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
 
    # Creating plot
    bp = ax.boxplot(data)    


    major_y_ticks = np.arange(0, max(max(data[0]),max(data[1]),max(data[2]))*1.05, 1)
    minor_y_ticks = np.arange(0, max(max(data[0]),max(data[1]),max(data[2]))*1.05, 0.2)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)
    ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8)
    ax.set_title("StepTime with tolerance 0,1,2 for client"+str(clientIdx))
    
    fig.savefig("./results/"+metrics_file+"_"+current_function_name+".png")
    plt.show()      


    print("DONE_DISPLAY")


if __name__ == "__main__":

    metrics_file1 = "RL_Metrics_E20_I25_B100_D50000_L0.01_M30_T0"
    # metrics_file2 = "RL_Metrics_E1_I5_B10_D50000"
    # metrics_file3 = "RL_Metrics_E1_I5_B200_D5000"
    # metrics_file4 = "RL_Metrics_E1_I5_B200_D50000"
    # with open("./results/"+metrics_file1+".pkl", 'rb') as f:
    #     RL_res1 = pickle.load(f)
    # with open("./results/"+metrics_file2+".pkl", 'rb') as f:
    #     RL_res2 = pickle.load(f)
    # with open("./results/"+metrics_file3+".pkl", 'rb') as f:
    #     RL_res3 = pickle.load(f)
    # with open("./results/"+metrics_file4+".pkl", 'rb') as f:
    #     RL_res4 = pickle.load(f)

    
    # display_split_layer_by_episode(RL_res1)
    # display_steps_and_relativeTime_per_episode(RL_res1)
    # display_eachStep_rew_maxIterTime_stepTime(RL_res1)
    # display_maxIterTime(RL_res1)
    # display_server_and_client_steptime(RL_res1)
    # display_server_and_client_maxSteptime(RL_res1)
    # ####
    # display_server_and_client_idle_time(RL_res1)
    # display_server_and_client_maxNmin_idle_time(RL_res1)
    # display_server_client_stepNidle_time(RL_res1)
    # display_1st_client_splitLayers_and_idle_time(RL_res1)
    # display_add_client_interstep_time(RL_res1)
    ####v entropy v#
    # display_entropyOnAvg_batches(RL_res1) #Also RL_res2
    # display_entropyAggregated_batches(RL_res1) #Also RL_res2
    # display_entropy_batches_avg_vs_aggregated(RL_res1) #Also RL_res2
    # display_surprise_vs_stepTime_ALL_splitLayer(RL_res1)   ##IMPORTANT ; Good find
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res1,0)
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res1,3)
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res2,0)
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res2,3)
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res3,0)
    # display_surprise_vs_stepTime_splitLayer_SingleClient(RL_res3,3)

    #display_boxplot_tolerance_vs_steptime_SingleClient(RL_res1,RL_res2,RL_res3,0)
    ### Maybe also do it for idle time? ^
    # print("AVG:  TOTAL_T \tSTEP_T \t\tCPU_W \t\tRAM_W \t\tREWARDS \t### RUN IDENTIFIER")
    # simple_print_avg_objectives(RL_res1,metrics_file1[11:])
    # simple_print_avg_objectives(RL_res2,metrics_file2[11:])
    # simple_print_avg_objectives(RL_res3,metrics_file3[11:])
    # simple_print_avg_objectives(RL_res4,metrics_file4[11:])
    
    #iterate_and_process_ALL_RUNS()
    
    #Variables to consider Episodes nr, iteration number, batch_size, Datalenght_size, 
    print("FINIIIIIISH")