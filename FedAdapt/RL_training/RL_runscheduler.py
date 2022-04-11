
import sys
sys.path.append('../FedAdapt')

import config
from RL_serverrun import server_main
from RL_clientrun import client_main
import time
import psutil

def objective1():
    
    #Episode, Tolerance, Update_epochs?, Steps?, Iter
    print("########################## \nOBJECTIVE 1")
    episode_range = [3]#[1,5,10,25,50] # 5
    iteration_range = [5] #[1,3,5,10,20] # 5
    batch_size = [10,200] #[1,10,50,100,200] #5
    data_lenght = [50000,5000] # [50000,25000,15000,5000, 2500] #5
    
    ################
    ##################
    psutil.cpu_percent()
    for e in episode_range:
        config.max_episodes = e
        for i in iteration_range:
            new_iter_dict = {x: i for x in config.iteration} #Changes all the values in dict to the integer i
            config.iteration = new_iter_dict
            for b in batch_size:
                config.B = b
                for d in data_lenght:


            
                    try:
                        time_server_start = time.perf_counter()
                        print("##########################\nRUN METRICS: \n  E: "+str(e)+" \n  I: "+str(i)+ "\n  B: "+str(b)+" \n  D: "+str(d)+"\n##########################")   
                        #######################             
                        run_identifier = "E"+str(e)+"_I"+str(i)+"_B"+str(b)+"_D"+str(d)

                        ####################
                        start_run(run_identifier)
                        
                    except Exception as exception:
                        print("EXCEPTION OCCURED DURING RUN: E "+str(e)+" ,i "+str(i))
                        print(exception)
                        ###TODO: Try with clients; check if server start/ipaddress assignation is a problem

                    
                    #print("CPU_PERCENT_RUN: "+ str(psutil.cpu_percent()))
                    time_server_finish = time.perf_counter()
                    print("RUN TIME: "+ str(time_server_finish-time_server_start))
                    print("Waiting 5s for address deallocation...")
                    time.sleep(5) #waiting for de-allocation of server address
                    print("Finished run")

def objective2():
    #Reward, etc??
    print("OBJECTIVE 2")

def start_run(run_identifier):
    device_type = sys.argv[1]
    if device_type == "server":
        server_main(run_identifier)
    if device_type == "client":
        client_main()


if __name__ == "__main__": 
    objective1()
    #objective2()
    

