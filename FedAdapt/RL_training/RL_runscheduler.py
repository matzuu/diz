
import sys
sys.path.append('../FedAdapt')

import config
from RL_serverrun import server_main
from RL_clientrun import client_main
import time

def objective1():
    #Episode, Tolerance, Update_epochs?, Steps?, Iter
    print("########################## \nOBJECTIVE 1")
    episode_range = [1,2,3]
    iteration_range = [3,5]
    for e in episode_range:
        config.max_episodes = e
        for i in iteration_range:
            new_iter_dict = {x: i for x in config.iteration} #Changes all the values in dict to the integer i
            config.iteration = new_iter_dict
            time_server_start = time.perf_counter()
            try:
                
                print("##########################\nRUN METRICS: \n  E: "+str(e)+" \n  I: "+str(i)+"\n##########################")                
                start_run()
            except:
                print("EXCEPTION OCCURED DURING RUN: E "+str(e)+" ,i "+str(i))

                ###TODO: Try with clients; check if server start/ipaddress assignation is a problem

            time_server_finish = time.perf_counter()
            print("RUN TIME: "+ str(time_server_finish-time_server_start))

def objective2():
    #Reward, etc??
    print("OBJECTIVE 2")

def start_run():
    device_type = sys.argv[1]
    if device_type == "server":
        server_main()
    if device_type == "client":
        client_main()


if __name__ == "__main__": 
    objective1()
    #objective2()
    

