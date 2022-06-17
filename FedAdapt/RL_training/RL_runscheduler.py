
import sys
sys.path.append('../FedAdapt')

import config
from RL_serverrun import server_main
from RL_clientrun import client_main
import time
import psutil
import os

def scope1():

    device_type = sys.argv[1]
    
    #Episode, Tolerance, Update_epochs?, Steps?, Iter
    print("########################## \nStarted run scheduler. The runs will take a while...")
    #Stoped at ep 10 and iter 50
    episode_range = [30] #DEF [10,30]#    3~4min/run for 10 epis  
    iteration_range = [5,25] #DEF [5,25]  #REDO WITH 25,50,100
    batch_size_range = [50,100] #DEF [50,100] ##
    data_lenght_range = [10000,50000]# DEF[10000,50000] #Cannot do 1000
    learning_rate_range = [0.001, 0.002, 0.02, 0.05, 0.1, 0.2]# DEF[0.005,0.01] ##REDO 0.001, 0.002, 0.02 0.05 0.1 0.2
    max_update_epochs_range = [10,30] # DEF[10,30] ##Done 100
    tolerance_range = [0] #[0,1,2] # 3   ##DEF [0]
    print("Total number of expected benchmark runs: " + str(len(episode_range)*len(iteration_range)*len(batch_size_range)*len(data_lenght_range)*len(learning_rate_range)*len(max_update_epochs_range)*len(tolerance_range)))
    time.sleep(2)
    ################
    ##################
    psutil.cpu_percent()
    for e in episode_range:
        config.max_episodes = e
        for i in iteration_range:
            new_iter_dict = {x: i for x in config.iteration} #Changes all the values in dict to the integer i
            config.iteration = new_iter_dict
            for b in batch_size_range:
                ###########################
                                 
                config.B = b
                for d in data_lenght_range:
                    config.N = d

                    ##MAX ITERATION NUMBER = DATASET_SIZE (d) / ( NUMBER OF DEVICES (K) * BATCH_SIZE (B) ) i.e 5000 / (5 * 100) = 10 Due to implementation issues with trainloader
                    max_iter_number = d / ( 5 * b )
                    if i <= max_iter_number:
                        for l in learning_rate_range:
                            config.LR = l
                            for m in max_update_epochs_range:
                                config.max_update_epochs = m
                                for t in tolerance_range:
                                    config.tolerance_counts = t

                                    try:
                                        if(m * e <= 1000 or 1>0): #want to gurantee a certain amount of steps to get metrics from
                                            time_server_start = time.perf_counter()
                                            print("##########################\nRUN METRICS: \n  E: "+str(e)+" \n  I: "+str(i)+ "\n  B: "+str(b)+" \n  D: "+str(d)+" \n  L: "+str(l)+ "\n  M: "+str(m)+" \n  T: "+str(t)+"\n##########################")   
                                            #######################             
                                            run_identifier = "E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t)
                                            try:
                                                os.remove("PPO.pth") ##Remove old trained model, so it creates a new one,and trains without being influenced by previous trains
                                            except Exception as exception_file_already_removed:
                                                pass #file already removed

                                            ####################
                                            start_run(run_identifier,device_type)
                                        
                                    except Exception as exception:
                                        print("EXCEPTION OCCURED DURING RUN: E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t))
                                        print(exception)
                                        ###TODO: Try with clients; check if server start/ipaddress assignation is a problem

                                    
                                    #print("CPU_PERCENT_RUN: "+ str(psutil.cpu_percent()))
                                    time_server_finish = time.perf_counter()
                                    print("FINISHED RUN: " + run_identifier)
                                    print("RUN TIME: "+ str(time_server_finish-time_server_start))
                                    if device_type == "server":
                                        print("Waiting 5s for address deallocation...")
                                        time.sleep(5) #waiting for de-allocation of server address
                                    elif device_type == "client":
                                        print("Waiting 8s for address deallocation...")
                                        time.sleep(8) #waiting for de-allocation of server address
                                

def scope2():
    print("Started scope 2")
    device_type = sys.argv[1]
    reliability_runs = 4
    
    variable_name_list = ['max_episodes',
                   'max_iterations' ,
                   'batch_size',
                   'datasize_lenght',
                   'learning_rate',
                   'max_update_epochs'] 
    variables_list = [[10, 2, 459, 1000, 2, 5],[30, 2, 459, 1000, 21, 5],[30, 2, 403, 1000, 2, 5],[30, 2, 459, 1000, 3, 5],[7, 2, 30, 1099, 5, 5],[30, 2, 30, 1885, 5, 5],[28, 2, 30, 3031, 5, 5],[6, 2, 30, 4786, 5, 5],[4, 1, 30, 2656, 5, 5],[42, 1, 30, 4007, 5, 5],[8, 1, 30, 16991, 5, 5],[5, 1, 30, 23879, 5, 5],[4, 1, 30, 25052, 5, 5]]

    print("Total number of expected benchmark runs: " + str(reliability_runs * len(variables_list)))
    time.sleep(2)
    ################
    ##################
    psutil.cpu_percent()

    for combination in variables_list:
        for idx_r in range(reliability_runs):
            config.max_episodes = combination[0]
            new_iter_dict = {x: combination[1] for x in config.iteration} #Changes all the values in dict to the integer i
            config.iteration = new_iter_dict
            config.B = combination[2]
            config.N = combination[3]
            config.LR = combination[4]/1000 
            config.max_update_epochs = combination[5]
            run_identifier = "MOO_E"+str(combination[0])+"_I"+str(combination[1])+"_B"+str(combination[2])+"_D"+str(combination[3])+"_L"+str(combination[4]/1000)+ "_M"+str(combination[5])+"_R"+str(idx_r)
            print("Run metrics "+ run_identifier)
            
            time_server_start = time.perf_counter()            
            try:
                os.remove("PPO.pth") ##Remove old trained model, so it creates a new one,and trains without being influenced by previous trains
            except Exception as exception_file_already_removed:
                pass #file already removed            
            try:
                start_run(run_identifier,device_type)
            except Exception as exception:
                print("EXCEPTION OCCURED DURING RUN:" + run_identifier)
                print("##" + str(exception))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

            time_server_finish = time.perf_counter()
            print("FINISHED RUN: " + run_identifier)
            print("RUN TIME: "+ str(time_server_finish-time_server_start))
            if device_type == "server":
                print("Waiting 5s for address deallocation...")
                time.sleep(5) #waiting for de-allocation of server address
            elif device_type == "client":
                print("Waiting 8s for address deallocation...")
                time.sleep(8) #waiting for de-allocation of server address

    return

def scope3():
    print("Started scope 3: Default Params")
    device_type = sys.argv[1]
    reliability_runs = 10
    variable_name_list = ['max_episodes',
                   'max_iterations' ,
                   'batch_size',
                   'datasize_lenght',
                   'learning_rate',
                   'max_update_epochs'] 
    

    
    for idx_r in range(10,reliability_runs+10):
        run_identifier = "BASE_E100_I5_B100_D50000_LR0.01_M10_R"+str(idx_r)
        print("Run metrics "+ run_identifier)
        
        time_server_start = time.perf_counter()            
        try:
            os.remove("PPO.pth") ##Remove old trained model, so it creates a new one,and trains without being influenced by previous trains
        except Exception as exception_file_already_removed:
            pass #file already removed            
        try:
            start_run(run_identifier,device_type)
        except Exception as exception:
            print("EXCEPTION OCCURED DURING RUN:" + run_identifier)
            print("##" + str(exception))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        time_server_finish = time.perf_counter()
        print("FINISHED RUN: " + run_identifier)
        print("RUN TIME: "+ str(time_server_finish-time_server_start))
        if device_type == "server":
            print("Waiting 5s for address deallocation...")
            time.sleep(5) #waiting for de-allocation of server address
        elif device_type == "client":
            print("Waiting 8s for address deallocation...")
            time.sleep(8) #waiting for de-allocation of server address

    return

def start_run(run_identifier,device_type):
    
    if device_type == "server":
        server_main(run_identifier)
    if device_type == "client":
        client_main()


if __name__ == "__main__": 
    #scope1()
    scope2()
    #scope3()

    print("DONE")
    

