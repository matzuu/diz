import pickle #
import matplotlib.pyplot as plt
import numpy as np
import operator
import inspect
import sys
sys.path.append('../FedAdapt')
import config
import processing.entropy as entro
import pandas as pd

def simple_print_avg_objectives(RL_res1,run_identifier):

    step_counter = 0
    train_time_list = []
    resource_wastage_cpu_list = []
    resource_wastage_ram_list = []
    reward_list = []

    for episode_index in (range(1,len(RL_res1)-3)): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)
        
        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        
        for step_index in range(len(episode)-4): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_counter +=1
            train_time_list.append(step["server_step_time_total"])
            resource_wastage_cpu_list.append(step['cpu_wastage'])
            resource_wastage_ram_list.append(step['ram_wastage'])
            reward_list.append(step['rewards'])


    #print("##################################################################### "+ run_identifier)
    total_run_time = format(RL_res1["RL_time_total_server"], '.3f')
    avg_train_time = format((sum(train_time_list)/len(train_time_list)), '.3f')
    avg_cpu_wastage = format((sum([sum(item.values()) for item in resource_wastage_cpu_list])/len(resource_wastage_cpu_list)), '.3f')
    avg_ram_wastage = format((sum([sum(item.values()) for item in resource_wastage_ram_list])//len(resource_wastage_ram_list) ), '.3f')
    avg_rewards = format((sum(reward_list)/len(reward_list) ), '.3f') ###REWARDS AS IMPROVEMENT OVERALL, RATHER THAN LAST STEP? ==> IMPROVEMENT CALCULATED AS 
    print("   :  "+ total_run_time+ "\t"+avg_train_time+ "\t\t" + avg_cpu_wastage + "\t\t" +avg_ram_wastage + "\t\t" +avg_rewards +"\t\t### "+ run_identifier)
    return

def iterate_and_process_ALL_RUNS():
    #Make it the same as the RL_runscheduler.py
    episode_range = [1,50]#[1,5,10,25,50] # 5
    iteration_range = [5,50] #[1,3,5,10,20] # 5
    batch_size_range = [10,200] #[1,10,50,100,200] #5
    data_lenght_range = [5000,50000]# [50000,25000,15000,5000, 2500] #5
    learning_rate_range = [0.005,0.03]
    max_update_epochs_range = [5,50]
    tolerance_range = [0,2]

    whole_df = pd.DataFrame()
    
    for e in episode_range:
        config.max_episodes = e
        for i in iteration_range:
            new_iter_dict = {x: i for x in config.iteration} #Changes all the values in dict to the integer i
            config.iteration = new_iter_dict
            #print("AVG:  TOTAL_T \t STEP_T \t CPU_W \t\t RAM_W \t\t REWARDS \t### RUN IDENTIFIER")
            print("running...")
            for b in batch_size_range:
                config.B = b
                for d in data_lenght_range:
                    for l in learning_rate_range:
                        config.LR = l
                        for m in max_update_epochs_range:
                            config.max_update_epochs = m
                            for t in tolerance_range:
                                config.tolerance_counts = t

                                try:
                                    run_identifier = "E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t)
                                    #print("#############################################\nRUN METRICS: " + run_identifier) 
                                    
                                    metrics_file = "RL_Metrics_" +str(run_identifier)
        
                                    with open("./results/"+metrics_file+".pkl", 'rb') as f:
                                        RL_res1 = pickle.load(f)

                                    #simple_print_avg_objectives(RL_res1,run_identifier)
                                    current_run_df = get_step_objectives_from_run(RL_res1,e,i,b,d,l,m,t)
                                    whole_df = pd.concat([whole_df,current_run_df],ignore_index= True)
                                    #print(whole_df)
                                

                                except Exception as exception:
                                    print("EXCEPTION OCCURED DURING PRINT: E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t))
                                    print(exception)

    print(whole_df)
    return

def get_step_objectives_from_run(RL_res1,max_episodes,max_iterations,batch_size,data_lenght,learning_rate,max_update_epochs,tolerance):

    step_counter = 0
    step_time_list = []
    res_wastage_list = []
    rewards_list = []
    split_layer_list = []

    for episode_index in (range(1,len(RL_res1)-3)): #index from 1 to 100; (Len of RL_res1 is 101, but contains 100 episodes + 1 time_value) (episodes start at 1)
        
        episode = RL_res1["episode_"+str(episode_index)] #Get Episode value
        
        for step_index in range(len(episode)-4): # (steps start at 0) (1 value in dict is the total episode time; so -1 at len)
            step = episode["step_"+str(step_index)]
            step_counter +=1
            step_time_list.append(step["server_step_time_total"])
            current_cpu_wastage = sum(list(step['cpu_wastage'].values()))
            current_ram_wastage = sum(list(step['ram_wastage'].values()))
            res_wastage_list.append( current_cpu_wastage + current_ram_wastage)
            rewards_list.append(step['rewards'])
            split_layer_list.append(step['split_layer'])
  
    max_episodes_list = [max_episodes] * step_counter
    max_iterations_list = [max_iterations]  * step_counter
    batch_size_list = [batch_size]  * step_counter
    data_lenght_list = [data_lenght]  * step_counter
    learning_rate_list = [learning_rate]  * step_counter
    max_update_epochs_list = [max_update_epochs]  * step_counter
    tolerance_list = [tolerance]  * step_counter

    current_run_df = pd.DataFrame({'train_times': step_time_list,
                   'resource_wastages': res_wastage_list,
                   'rewards': rewards_list,
                   'offload_configuration': split_layer_list , #TODO,1 element is currently in form of List[A,B,C,D,E] should i change it to 5 variables i.e offload_client1 : 1, offload_client2 : 5, ...
                   'max_episodes': max_episodes_list ,
                   'max_iterations': max_iterations_list  ,
                   'batch_size': batch_size_list ,
                   'data_lenght': data_lenght_list ,
                   'learning_rate': learning_rate_list ,
                   'max_update_epochs': max_update_epochs_list ,
                   'tolerance': tolerance_list          
                   })

    return current_run_df


if __name__ == "__main__":
    iterate_and_process_ALL_RUNS()

    #128 runs, 6529 steps
    print("FINIISH")