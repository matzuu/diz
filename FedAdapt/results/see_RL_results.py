import pickle #
import matplotlib.pyplot as plt
import numpy as np
##IMPORTS##
#########################################




def display_split_layer_by_episode(RL_res1):
    split_layer_matrix = []
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
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
    ax.grid()

    fig.savefig("test.png")
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


    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)
        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        episode_nrSteps_list.append(len(episode)-1)
        episode_avg_step_time.append(episode["episode_time_total"]/len(episode)-1)

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
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 17, 2)
    minor_y_ticks = np.arange(0, 17, 1)
    
    ###########################################################
    ##No need to change bellow   
    
    ax.set_xticks(major_x_ticks)
    ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.8)
    
    ax.legend(loc="upper right")
    fig.savefig("test.png")
    plt.show()


    print("DONE_DISPLAY")

def display_eachStep_rew_maxIterTime_stepTime(RL_res1):
    step_reward_list = []
    step_maxIter_list = []
    step_totalTimeServer_list = []
    episode_stepsNr_indexes = [0]
    nrSteps = 0
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
    episode_stepsNr_indexes.append(nrSteps+1)
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
    fig.savefig("test.png")
    plt.show()


    print("DONE_DISPLAY")

def display_maxIterTime(RL_res1):
    step_maxIter_list = []
    episode_stepsNr_indexes = [0]
    nrSteps = 0
    for episode_index in (range(1,len(RL_res1))): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)

        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        nrSteps += len(episode)-1
        episode_stepsNr_indexes.append(nrSteps) #used for displaying when an episode ends on the plot

        for step_index in range(len(episode)-1): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_maxIter_list.append(step["maxtime_iteration"])

    ##########################################################
    x = range(1,len(step_maxIter_list)+1)
    y2 = np.array(step_maxIter_list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y2, color='tab:red',label="Max Iteration Time")
    ax.set(xlabel='Step Nr', ylabel=' Time (seconds)',
        title='Max Iteration Time')

    # Major ticks every 20, minor ticks every 5
    major_x_ticks = np.arange(0, 101, 5)
    minor_x_ticks = np.arange(0, 101, 1)
    major_y_ticks = np.arange(0, 1.1, 0.25)
    minor_y_ticks = np.arange(0, 1.1, 0.1)
    episode_stepsNr_indexes.append(nrSteps+1)
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
    fig.savefig("test.png")
    plt.show()


    print("DONE_DISPLAY")

if __name__ == "__main__":
    with open('./results/RL_Metrics1.pkl', 'rb') as f:
        RL_res1 = pickle.load(f)
    #display_steps_and_relativeTime_per_episode(RL_res1)
    #display_eachStep_rew_maxIterTime_stepTime(RL_res1)
    display_maxIterTime(RL_res1)
    print("Loaded Metrics dataset")