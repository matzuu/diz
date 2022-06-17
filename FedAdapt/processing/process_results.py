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
import os

def simple_get_and_print_real_data_avg_objectives(RL_res1,run_identifier):

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
    avg_cpu_wastage = format((sum([sum(item.values()) for item in resource_wastage_cpu_list])/len(resource_wastage_cpu_list)), '.2f')
    avg_ram_wastage = format((sum([sum(item.values()) for item in resource_wastage_ram_list])//len(resource_wastage_ram_list) ), '.2f')
    avg_rewards = format((sum(reward_list)/len(reward_list) ), '.3f') ###REWARDS AS IMPROVEMENT OVERALL, RATHER THAN LAST STEP? ==> IMPROVEMENT CALCULATED AS 

    #print("AVG:  TOTAL_T \tSTEP_T \t\tCPU_W \t\tRAM_W \t\tREWARDS \t### RUN IDENTIFIER")
    #print("   :  "+ total_run_time+ "\t"+avg_train_time+ "\t\t" + avg_cpu_wastage + "\t\t" +avg_ram_wastage + "\t\t" +avg_rewards +"\t\t### "+ run_identifier)
    return float(avg_train_time),float(avg_rewards), round((float(avg_cpu_wastage)+float(avg_ram_wastage)),3)
def create_df_from_X_RUNS(folder = "metrics", max_runs = 100000):
    import re

    whole_df = pd.DataFrame()

    file_counter = 0

    total_files = str(len(os.listdir('./results/' + folder +"/")))
    print ("## Processed " + str(file_counter) +" files out of " + total_files)

    for file_name in os.listdir('./results/' + folder+"/"):
        try:
            with open("./results/"+ folder+"/" + file_name, 'rb') as f:
                RL_res1 = pickle.load(f)

            #simple_print_avg_objectives(RL_res1,run_identifier)

            e,i,b,d,l,m,t = re.findall(r'[0]\.\d+|[\d]+',file_name) #t - tolerance is no longer used
            current_run_df = get_step_metrics_from_run(RL_res1,e,i,b,d,l,m)
            whole_df = pd.concat([whole_df,current_run_df],ignore_index= True)
            file_counter +=1
            if file_counter%100 == 0:
                print("Processed "+str(file_counter) + " files out of "+total_files)
            #print(whole_df)
        

        except Exception as exception:
            print("EXCEPTION OCCURED DURING PRINT FOR FILE: " + file_name)
            print(exception)

        if file_counter > max_runs:
            return whole_df    


    return whole_df
def get_step_metrics_from_run(RL_res1,max_episodes,max_iterations,batch_size,datasize_lenght,learning_rate,max_update_epochs):

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
            step_time_list.append(round(step["server_step_time_total"],4))
            current_cpu_wastage = sum(list(step['cpu_wastage'].values()))
            current_ram_wastage = sum(list(step['ram_wastage'].values()))
            res_wastage_list.append( round((current_cpu_wastage + current_ram_wastage),4))
            rewards_list.append(round(step['rewards'],4))
            split_layer_list.append(step['split_layer'])
  
    max_episodes_list = [max_episodes] * step_counter
    max_iterations_list = [max_iterations]  * step_counter
    batch_size_list = [batch_size]  * step_counter
    datasize_lenght_list = [datasize_lenght]  * step_counter
    learning_rate_list = [learning_rate]  * step_counter
    max_update_epochs_list = [max_update_epochs]  * step_counter
    #tolerance_list = [tolerance]  * step_counter

    current_run_df = pd.DataFrame({'train_times': step_time_list,
                   'resource_wastages': res_wastage_list,
                   'rewards': rewards_list,
                   'offload_configuration': split_layer_list , #TODO,1 element is currently in form of List[A,B,C,D,E] should i change it to 5 variables i.e offload_client1 : 1, offload_client2 : 5, ...
                   'max_episodes': max_episodes_list ,
                   'max_iterations': max_iterations_list  ,
                   'batch_size': batch_size_list ,
                   'datasize_lenght': datasize_lenght_list ,
                   'learning_rate': learning_rate_list ,
                   'max_update_epochs': max_update_epochs_list 
                   #'tolerance': tolerance_list          
                   })

    return current_run_df
def create_file_DF_from_ALL_metrics(filename = "Aggregated_panda_DF",folder = "metrics"):
    whole_df = create_df_from_X_RUNS(folder)
    file_path = config.home + './results/'+ filename +'.pkl'
    whole_df.to_pickle(file_path)
    print(whole_df)

def create_file_small_DF_from_metrics(filename = "small_panda_DF"):
    whole_df = create_df_from_X_RUNS(20)
    file_path = config.home + './results/'+ filename +'.pkl'
    whole_df.to_pickle(file_path)
    print(whole_df)

def get_df_from_file(filepath = './results/Aggregated_panda_DF.pkl'):
    try:
        return pd.read_pickle(filepath)
    except:
        return pd.DataFrame()
def get_hyperparams_metrics_from_df_file(filepath = './results/Aggregated_panda_DF.pkl'): ####NOT USED?
    
    df = get_df_from_file(filepath)

    offload_configuration_l = df.loc[:,"offload_configuration"].tolist()
    max_episodes_l = df.loc[:,"max_episodes"].tolist()
    max_iterations_l = df.loc[:,"max_iterations"].tolist()
    batch_size_l = df.loc[:,"batch_size"].tolist()
    datasize_lenght_l = df.loc[:,"datasize_lenght"].tolist()
    learning_rate_l = df.loc[:,"learning_rate"].tolist()
    max_update_epochs_l = df.loc[:,"max_update_epochs"].tolist()
    #tolerance_l = df.loc[:,"tolerance"].tolist()
    return offload_configuration_l,max_episodes_l,max_iterations_l,batch_size_l,datasize_lenght_l,learning_rate_l,max_update_epochs_l#,tolerance_l
def visualize_boxplots_of_objective_base_on_all_variables(df,objective,variable_name_list):
    for variable in variable_name_list:
        visualize_boxplots_of_objective_based_on_variable(df,objective,variable)
        print("Visualizing: " + variable)
def visualize_boxplots_of_objective_based_on_variable(df,objective,variable):

    data_to_display = []
    tick_label_list = []
    
    metrics_dict = get_values_of_column1_based_on_column2_criteria(df,objective,variable)

    #Sort the dict
    float_vars_list = ['learning_rate','hypervolume','avg_crowding_distance','mutation_probability','mutation_dist_idx','crossover_probability','crossover_dist_idx']
    if(variable in float_vars_list): 
        metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: float(x[0]))) #float values
    else: #variable isn't float        
        metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: int(x[0]))) #int values


    for key,valueList in metrics_dict.items():
        tick_label_list.append(str(key))
        data_to_display.append(valueList)
    
    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance    
    ax = fig.add_subplot(111)
    
    # Creating plot

    bp = ax.boxplot(data_to_display)

    ax.set_xticklabels(tick_label_list)
    ax.set_xlabel(variable)
    ax.set_ylabel(objective)

    ax.grid()
    
    # show plot
    plt.show()
def get_values_of_column1_based_on_column2_criteria(df: pd.DataFrame,column1_name: str,column2_name: str)-> dict:

    column2_value_range = df[column2_name].unique()

    return_dict = dict()
    for value in column2_value_range:

        values_list = df.loc[df[column2_name] == value,column1_name].tolist()
        return_dict[value] = values_list
        
    return return_dict
def get_avg_of_column1_based_on_column2and3(df: pd.DataFrame,column1_name: str,column2_name: str,column3_name: str)-> dict:  ##Not used?

    column2_value_range = df[column2_name].unique()
    column3_value_range = df[column3_name].unique()

    print("Avg / Median Value of: column1_name based on: "+ column2_name + " " + column3_name)
    for value_2 in column2_value_range:

        for value_3 in column3_value_range:
            sub_df = df[(df[column2_name] == value_2 ) & (df[column3_name] == value_3 )]
            #print(sub_df)
            med_value =  sub_df[column1_name].median()
            print(value_2 +" "+ value_3 + " Median Train Time:  " + str(med_value))
        
    print("DONE")

def get_correlation_of_objective_for_all_variables(df,objective,variable_name_list):
    
    #print("###########################################")
    corr_array = []
    for variable in variable_name_list:        
        column1 = np.array(df[variable].astype(float))
        column2 = np.array(df[objective])
        r_corr = np.corrcoef(column1, column2)
        #print("Correlation between   " + objective + " " + variable +":   \t" + str(r_corr[0][1]))
        #print(r_corr)
        corr_array.append(r_corr[0][1])
    return corr_array
    
def get_interp_function_for_obj_and_variable(df,objective,variable):
    from scipy.interpolate import interp1d

    metrics_dict = get_values_of_column1_based_on_column2_criteria(df,objective,variable)

    #Sort the dict
    if(variable != "learning_rate"): 
        metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: int(x[0]))) #int values
        X_variable_vals =  list(map(int, metrics_dict.keys()))
    else:
        metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: float(x[0]))) #float values
        X_variable_vals =  list(map(float, metrics_dict.keys()))

    Y_obj_expected_values = [round((sum(val)/len(val)),4) for val in metrics_dict.values()]
    ############################################
    ### Gather Data ^ | v Interpolate ##########
    ############################################

    f = interp1d(X_variable_vals, Y_obj_expected_values)
    #f2 = interp1d(X_variable_vals, Y_obj_expected_values, kind='cubic') #NEED AT LEAST 4 POINTS

    return f,f #f2 ## Call f(x) like in math to get result y i.e f(13) = 5.67

def interpolate_and_plot_all_variables(df,objective,variable_name_list):
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    print("###########################################")
    print("Interpolating Variables for obj: " + str(objective))
    for variable in variable_name_list:        

        ##########################################
        metrics_dict = get_values_of_column1_based_on_column2_criteria(df,objective,variable)
        #Sort the dict
        if(variable != "learning_rate"): 
            metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: int(x[0]))) #int values
            X_variable_vals =  list(map(int, metrics_dict.keys()))
        else:
            metrics_dict= dict(sorted(metrics_dict.items(), key=lambda x: float(x[0]))) #float values
            X_variable_vals =  list(map(float, metrics_dict.keys()))

        Y_obj_expected_values = [round((sum(val)/len(val)),4) for val in metrics_dict.values()]
        
        
        f = interp1d(X_variable_vals, Y_obj_expected_values)
        f2 = interp1d(X_variable_vals, Y_obj_expected_values, kind='cubic') #NEED AT LEAST 4 POINTS
        
        xnew = np.linspace(min(X_variable_vals), max(X_variable_vals), num=50, endpoint=True)
        ##############################################
        ##PLOTTING############

        fig = plt.figure(figsize =(10, 7))
    
        # Creating axes instance    
        ax = fig.add_subplot(111)
        ax.plot(X_variable_vals, Y_obj_expected_values, 'o', xnew, f(xnew), '-', xnew , f2(xnew), '--')

        #ax.set_xticklabels()
        ax.set_xlabel(variable)
        ax.set_ylabel(objective)

        ax.grid()

        
        #plt.plot(X_variable_vals, Y_obj_expected_values, 'o', xnew, f(xnew), '-', xnew , f2(xnew), '--')
        #plt.legend(['data', 'linear'], loc='best')
        plt.legend(['data', 'linear', 'cubic'], loc='best') 
        plt.xlabel(variable)
        plt.ylabel(objective)
        plt.show()

def calculate_objectives_score(objective, corr_arr, f_list ,v_max_episodes = 10, v_max_iterations = 5, v_batch_size = 100, v_datasize_lenght = 50000, v_learning_rate = 0.01, v_max_update_epochs = 10) -> float:
    # variable_name_list =  ['max_episodes',
    #                'max_iterations' ,
    #                'batch_size',
    #                'datasize_lenght',
    #                'learning_rate',
    #                'max_update_epochs'] 
    # corr_arr = get_correlation_of_objective_for_all_variables(df,objective,variable_name_list)

    bias = [1,1,1,1,1,1]
    final_bias = 0
    if objective == "train_times":
        bias = [0.1, 10, 1, 1, 0.1, 0.1]
        final_bias = 0
    if objective == "rewards":
        bias = [0.1,10,5,1,1,0.5]
        final_bias = 0
    w_max_episodes = corr_arr[0] * bias[0]
    w_max_iterations = corr_arr[1] * bias[1]
    w_batch_size = corr_arr[2] * bias[2]
    w_datasize_lenght = corr_arr[3] * bias[3] 
    w_learning_rate = corr_arr[4] * bias[4]
    w_max_update_epochs = corr_arr[5] * bias[5]

    sum_weights = w_max_episodes + w_max_iterations + w_batch_size + w_datasize_lenght + w_learning_rate + w_max_update_epochs

    f_max_episodes = f_list[0]
    f_max_iterations = f_list[1]
    f_batch_size = f_list[2]
    f_datasize_lenght = f_list[3]
    f_learning_rate = f_list[4]
    f_max_update_epochs = f_list[5]

    score_episodes = round(w_max_episodes * f_max_episodes(v_max_episodes) / sum_weights , 4)
    score_iterations =round( w_max_iterations * f_max_iterations(v_max_iterations) / sum_weights , 4)
    score_batches = round(w_batch_size * f_batch_size(v_batch_size) / sum_weights / sum_weights , 4)
    score_datasize = round(w_datasize_lenght * f_datasize_lenght(v_datasize_lenght) / sum_weights , 4)
    score_learning_rate = round(w_learning_rate * f_learning_rate(v_learning_rate) / sum_weights , 4)
    score_epochs = round(w_max_update_epochs * f_max_update_epochs(v_max_update_epochs) / sum_weights , 4)

    score_obj = round((score_episodes + score_iterations + score_batches + score_datasize + score_learning_rate + score_epochs) + final_bias , 4)
    #print(objective + " variables:\t\t E"+ str(v_max_episodes) +" \t\t I"+ str(v_max_iterations) +" \t\t B"+ str(v_batch_size) +" \t\t D"+ str(v_datasize_lenght)+" \td L"+ str(v_learning_rate) +" \t\t U"+str(v_max_update_epochs))
    #print(objective + " score: " + str(score_obj) +  "  = \t(E)"+ str(score_episodes) +" + \t(I)"+ str(score_iterations) +" + \t(B)"+ str(score_batches) +" + \t(D)"+ str(score_datasize)+" + \t(L)"+ str(score_learning_rate) +" + \t(U)"+str(score_epochs))
    #print("------------------------------------------------------------------------------------------")
    return score_obj

def get_interpolating_functions_list(df,objective,variable_name_list):

    f_list = []
    for var in variable_name_list:
        f_lin,f_poly = get_interp_function_for_obj_and_variable(df,objective,var)
        f_list.append(f_lin)

    # f_max_episodes,_ = get_interp_function_for_obj_and_variable(df,objective,"max_episodes")
    # f_max_iterations,_ = get_interp_function_for_obj_and_variable(df,objective,"max_iterations")
    # f_batch_size,_ = get_interp_function_for_obj_and_variable(df,objective,"batch_size")
    # f_datasize_lenght,_ = get_interp_function_for_obj_and_variable(df,objective,"datasize_lenght")
    # f_learning_rate,_ = get_interp_function_for_obj_and_variable(df,objective,"learning_rate")
    # f_max_update_epochs,_ = get_interp_function_for_obj_and_variable(df,objective,"max_update_epochs")

    return f_list

def get_and_print_simple_objs_from_file(metrics_file1):

    try:
        with open("./results/metrics/"+metrics_file1+".pkl", 'rb') as f:
            RL_res1 = pickle.load(f)
        train_t,rewards,rs_wastage = simple_get_and_print_real_data_avg_objectives(RL_res1,metrics_file1[11:])   
    except:
        #print("File not found.... probably: " + metrics_file1)
        train_t,rewards,rs_wastage = -1000000.0,-10000000.0,-10000000.0

    return train_t,rewards,rs_wastage

def get_unique_hyperparams_df(df):
    print("UNIQUE VALUES")

    hyperparams_list = []
    for col in df:
        if(col != 'train_times' and col != 'rewards' and col != 'resource_wastages' and col !='offload_configuration'):
            aux_list = list(df[col].unique())
            if col == 'learning_rate':
                aux_list = [float(i) for i in aux_list]
            else:
                aux_list = [int(i) for i in aux_list]
            aux_list.sort()
            print(col + ":     \t" + str(aux_list))
            hyperparams_list.append(aux_list)

def print_real_vs_simulated_error(df,objective,variable_name_list):
    
    f_list = get_interpolating_functions_list(df,objective,variable_name_list)
    corr_arr = get_correlation_of_objective_for_all_variables(df,objective,variable_name_list)
    
    error_dict = dict()
    key_list = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100] 
    for key in key_list:
        
        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E"+ str(key) + "_I5_B100_D50000_L0.01_M10_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,key,5,100,50000,0.01,10)   
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("EP\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ##################################################
    error_dict = dict()
    key_list = [1, 2, 3, 5, 7, 10, 20, 25, 30, 40, 50, 75]
    for key in key_list:

        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E10_I"+str(key)+"_B100_D50000_L0.01_M10_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,10,key,100,50000,0.01,10)   
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("IT\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ###################################################

    error_dict = dict()
    key_list = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    for key in key_list:

        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E10_I5_B"+ str(key) + "_D50000_L0.01_M10_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,10,5,key,50000,0.01,10)   
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("BA\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ###################################################

    error_dict = dict()
    key_list = [1000, 5000, 10000, 25000, 50000]
    for key in key_list:

        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E10_I5_B100_D"+ str(key) + "_L0.01_M10_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,10,5,100,key,0.01,10)   
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("DS\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ###################################################

    error_dict = dict()
    key_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    for key in key_list:

        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E10_I5_B100_D50000_L"+ str(key) + "_M10_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,10,5,100,50000,key,10)   
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("LR\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ###################################################

    error_dict = dict()
    key_list = [1, 3, 5, 10, 15, 20, 30, 40, 50]
    for key in key_list:

        real_v_tt,real_v_rew,real_v_rw = get_and_print_simple_objs_from_file("RL_Metrics_E10_I5_B100_D50000_L0.01_M"+ str(key) + "_T0") 
        simu_v = calculate_objectives_score(objective,corr_arr,f_list,10,5,100,50000,0.01,key) 
        if objective == "train_times":
            real_v = real_v_tt
        elif objective == "rewards":
            real_v = real_v_rew
        else:
            real_v = real_v_rw
        if(abs(real_v - simu_v) < 100000):
            error_dict[key] = (round(abs(1 - (simu_v/real_v))*100,0),round( simu_v - real_v,3))
    print("UE\t: ERR%    \t: ERR Abs")
    for key,value in error_dict.items():
        print(str(key)+'\t: '+ str(value[0]) +'%    \t: '+str(value[1]) + 's')
    print("---------------------------------------------")
    ###################################################

def plot_simulated_obj_function(df,objective,variable_name_list):

    f_list = get_interpolating_functions_list(df,objective,variable_name_list)
    corr_arr = get_correlation_of_objective_for_all_variables(df,objective,variable_name_list)

    lower_bound = [1,1,1,1000,1,1]
    upper_bound = [100,75,500,50000,200,50]

    for var in variable_name_list:
        i = variable_name_list.index(var)
        x_list = range(lower_bound[i],upper_bound[i],max(1,int(upper_bound[i]/100)))
        if var == "learning_rate": 
            x_list = [elem/1000 for elem in x_list]
        y_list = []

        for x in x_list:
            if var == "max_episodes":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_max_episodes=x))
            elif var == "max_iterations":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_max_iterations=x))
            elif var == "batch_size":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_batch_size=x))
            elif var == "datasize_lenght":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_datasize_lenght=x))
            elif var == "learning_rate":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_learning_rate=x))
            elif var == "max_update_epochs":
                y_list.append(calculate_objectives_score(objective,corr_arr,f_list,v_max_update_epochs=x))
        #START PLOTTING  like X, F(X)  ; for x in variable names (ep,it,batch ...)

        fig = plt.figure(figsize =(10, 7))
    
        # Creating axes instance    
        ax = fig.add_subplot(111)
        ax.plot(x_list, y_list)

        #ax.set_xticklabels()
        ax.set_xlabel(var)
        ax.set_ylabel(objective)

        ax.grid()

        
        #plt.plot(X_variable_vals, Y_obj_expected_values, 'o', xnew, f(xnew), '-', xnew , f2(xnew), '--')
        #plt.legend(['data', 'linear'], loc='best')
        #plt.legend(['data', 'linear', 'cubic'], loc='best') 
        plt.xlabel(var)
        plt.ylabel(objective)
        plt.show()

    print("DONE")

def lots_of_plots(df,variable_name_list):

    plot_simulated_obj_function(df,"train_times",variable_name_list)
    plot_simulated_obj_function(df,"rewards",variable_name_list) #must provide big dataset
    plot_simulated_obj_function(df,"resource_wastages",variable_name_list)

    print_real_vs_simulated_error(df,"train_times",variable_name_list)    
    corr_arr = get_correlation_of_objective_for_all_variables(df,"train_times",variable_name_list)
    print("Corelation between train_times and variables:" + str(corr_arr))


    print_real_vs_simulated_error(df,"rewards",variable_name_list)    
    corr_arr = get_correlation_of_objective_for_all_variables(df,"rewards",variable_name_list)
    print("Corelation between rewards and variables:" + str(corr_arr))

    print_real_vs_simulated_error(df,"resource_wastages",variable_name_list)    
    corr_arr = get_correlation_of_objective_for_all_variables(df,"resource_wastages",variable_name_list)
    print("Corelation between resource_wastages and variables:" + str(corr_arr))


    get_correlation_of_objective_for_all_variables(df,"rewards",variable_name_list)
    get_correlation_of_objective_for_all_variables(df,"resource_wastages",variable_name_list)

    ##################################################
    interpolate_and_plot_all_variables(df,"train_times",variable_name_list)
    interpolate_and_plot_all_variables(df,"rewards",variable_name_list)
    interpolate_and_plot_all_variables(df,"resource_wastages",variable_name_list)

    visualize_boxplots_of_objective_base_on_all_variables(df,"train_times",variable_name_list)
    visualize_boxplots_of_objective_base_on_all_variables(df,"rewards",variable_name_list)
    visualize_boxplots_of_objective_base_on_all_variables(df,"resource_wastages",variable_name_list)

    print("finished plotting")
#def save_front_df(front,pop_size, mutation_prob, mutation_distr_i,crossover_prob,crossover_distr_i):
#################################################################################################
#####################################  MAIN  ####################################################
#################################################################################################
if __name__ == "__main__":
    #create_file_DF_from_ALL_metrics() #
    
    #create_file_small_DF_from_metrics()
    #create_file_DF_from_ALL_metrics("MOO_benchmark_DF","metrics_MOO_test") #TODO RUN#
    create_file_DF_from_ALL_metrics(filename="BASE_benchmark_DF",folder="metrics_RL_BASE") #TODO RUN#

    #df = get_df_from_file()
    #print(df)
    df_moo = get_df_from_file('./results/MOO_benchmark_DF.pkl')
    print(df_moo)
    df_base = get_df_from_file('./results/BASE_benchmark_DF.pkl')
    print(df_base)
    #df = get_df_from_file('./results/small_panda_DF.pkl')

    ##TESTING DF
    
    get_unique_hyperparams_df(df_moo)

    #print(df)

    variable_name_list = ['max_episodes',
                   'max_iterations' ,
                   'batch_size',
                   'datasize_lenght',
                   'learning_rate',
                   'max_update_epochs'] 

    #lots_of_plots(df,variable_name_list)


    ############################################################
    
    # corr_arr_train_times = get_correlation_of_objective_for_all_variables(df,"train_times",variable_name_list)
    # corr_arr_rewards = get_correlation_of_objective_for_all_variables(df,"rewards",variable_name_list)
    # corr_arr_resource_wastages = get_correlation_of_objective_for_all_variables(df,"resource_wastages",variable_name_list)

    # f_list_train_times = get_interpolating_functions_list(df,"train_times",variable_name_list)
    # f_list_rewards = get_interpolating_functions_list(df,"rewards",variable_name_list)
    # f_list_resource_wastages = get_interpolating_functions_list(df,"resource_wastages",variable_name_list)

    # simul_value = calculate_objectives_score("train_times",corr_arr_train_times,f_list_train_times, 
    #                                                             39,
    #                                                             2,
    #                                                             150,
    #                                                             1566,
    #                                                             0.005, #learning rate#
    #                                                             5)

    # simul_value2 = calculate_objectives_score("train_times",corr_arr_train_times,f_list_train_times, 
    #                                                             40,
    #                                                             1,
    #                                                             150,
    #                                                             24026,
    #                                                             0.005, #learning rate#
    #                                                             5)

    df_moo = df_moo.groupby(['max_episodes','max_iterations','batch_size','datasize_lenght','learning_rate','max_update_epochs'],as_index=False).mean()
    print(df_moo)
    #df_moo = df_moo.sort_values(by=['train_time'], ascending = False)
    df_base = df_base.groupby(['max_episodes','max_iterations','batch_size','datasize_lenght','learning_rate','max_update_epochs'],as_index=False).mean()
    print(df_base)
   
    print("FINIISH")