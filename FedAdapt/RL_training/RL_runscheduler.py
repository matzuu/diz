
import sys
sys.path.append('../FedAdapt')

import config
from RL_serverrun import server_main
from RL_clientrun import client_main
import time
import psutil
import os

def scope1():
    
    #Episode, Tolerance, Update_epochs?, Steps?, Iter
    print("########################## \nStarted run scheduler. The runs will take a while...")
    episode_range = [10]#,50,100] #[1,10,50,100] # 4 #Finished 1    3~4min/run for 10 epis
    iteration_range = [20] #[5,20,50,100] # 4 # Finished 5 #CONTINUE FOR 50,100
    batch_size_range = [50] #[10,50,100,200] #4 ##########RESET
    data_lenght_range = [5000,10000,25000,50000]# [5000,10000,25000,50000] #4 
    learning_rate_range = [0.005,0.01,0.03] # 3
    max_update_epochs_range = [5,10,50]  # 3
    tolerance_range = [0,1,2] # 3
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
                config.B = b
                for d in data_lenght_range:
                    config.N = d
                    for l in learning_rate_range:
                        config.LR = l
                        for m in max_update_epochs_range:
                            config.max_update_epochs = m
                            for t in tolerance_range:
                                config.tolerance_counts = t

                                try:
                                    time_server_start = time.perf_counter()
                                    print("##########################\nRUN METRICS: \n  E: "+str(e)+" \n  I: "+str(i)+ "\n  B: "+str(b)+" \n  D: "+str(d)+" \n  L: "+str(l)+ "\n  M: "+str(m)+" \n  T: "+str(t)+"\n##########################")   
                                    #######################             
                                    run_identifier = "E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t)
                                    try:
                                        os.remove("PPO.pth") ##Remove old trained model, so it creates a new one,and trains without being influenced by previous trains
                                    except Exception as exception_file_already_removed:
                                        pass #file already removed

                                    ####################
                                    start_run(run_identifier)
                                    
                                except Exception as exception:
                                    print("EXCEPTION OCCURED DURING RUN: E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)+"_L"+str(l)+ "_M"+str(m)+"_T"+str(t))
                                    print(exception)
                                    ###TODO: Try with clients; check if server start/ipaddress assignation is a problem

                                
                                #print("CPU_PERCENT_RUN: "+ str(psutil.cpu_percent()))
                                time_server_finish = time.perf_counter()
                                print("FINISHED RUN: " + run_identifier)
                                print("RUN TIME: "+ str(time_server_finish-time_server_start))
                                print("Waiting 5s for address deallocation...")
                                time.sleep(5) #waiting for de-allocation of server address
                                

def scope2():
    #Reward, etc??
    #do somethio
    return

def start_run(run_identifier):
    device_type = sys.argv[1]
    if device_type == "server":
        server_main(run_identifier)
    if device_type == "client":
        client_main()


if __name__ == "__main__": 
    scope1()
    #objective2()
    

