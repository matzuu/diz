import sys
import pandas as pd
sys.path.append('../FedAdapt')
import config

from process_results import get_df_from_file, get_correlation_of_objective_for_all_variables, visualize_boxplots_of_objective_based_on_variable


def run_MAIN_moo():

    from jmetal.core.problem import  IntegerProblem
    from jmetal.core.solution import  IntegerSolution
    from math import sqrt, exp, pow, sin

    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator import BitFlipMutation, SPXCrossover
    from jmetal.operator.mutation import CompositeMutation , PolynomialMutation , IntegerPolynomialMutation
    from jmetal.operator.crossover import CompositeCrossover ,SBXCrossover , IntegerSBXCrossover
    from jmetal.problem.multiobjective.unconstrained import MixedIntegerFloatProblem
    from jmetal.problem import ZDT1
    from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions
    from jmetal.util.termination_criterion import StoppingByEvaluations
    from jmetal.algorithm.multiobjective.smpso import SMPSO
    from jmetal.util.archive import CrowdingDistanceArchive
    from jmetal.core.quality_indicator import HyperVolume

    from process_results import get_df_from_file, calculate_objectives_score, get_interpolating_functions_list,get_correlation_of_objective_for_all_variables 
    #GLOBALS
    df = get_df_from_file()  #'./results/small_panda_DF.pkl'
    variable_name_list = ['max_episodes',
                   'max_iterations' ,
                   'batch_size',
                   'datasize_lenght',
                   'learning_rate',
                   'max_update_epochs'] 

    corr_arr_train_times = get_correlation_of_objective_for_all_variables(df,"train_times",variable_name_list)
    corr_arr_rewards = get_correlation_of_objective_for_all_variables(df,"rewards",variable_name_list)
    corr_arr_resource_wastages = get_correlation_of_objective_for_all_variables(df,"resource_wastages",variable_name_list)

    f_list_train_times = get_interpolating_functions_list(df,"train_times",variable_name_list)
    f_list_rewards = get_interpolating_functions_list(df,"rewards",variable_name_list)
    f_list_resource_wastages = get_interpolating_functions_list(df,"resource_wastages",variable_name_list)

    range_obj = [3] #[2,3]
    range_evals = [5000] #[1000,2000,5000,10000]
    range_pop_size = [100] # [100,200,300,500,1000] # 1500 takes 2x 1000
    range_mutation_p = [0.5,0.6,0.7,0.8,0.9] # [0.0,0.1,0.5,0.9,1.0]
    range_mutation_dist_i = [5.0,10.0] # [5.0,20.0,100.0,400.0]
    range_crossover_p = [0.8,1.0] # [0.8, 0.9, 1.0] [0.0,0.1,0.5,0.9,1.0]
    range_crossover_dist_i = [20.0,40.0,150.0,200.0] #[40.0+] [5.0,20.0,100.0,400.0]
    reliability_runs = 2

    total_runs =len(range_obj) * len(range_evals) * len(range_pop_size) * len(range_mutation_p) * len(range_mutation_dist_i) * len(range_crossover_p) * len(range_crossover_dist_i) * reliability_runs

    algorithm_name = "NSGAII"
    file_path = config.home + './results/df_MOO_'+ algorithm_name + '_3d6.pkl'
    moo_df = get_df_from_file(file_path)
    print(moo_df)
    #moo_df = pd.DataFrame()
    run_counter = 0

    

    class ZDT1_INT_TEST(IntegerProblem):
        """ Problem ZDT1.

        .. note:: Bi-objective unconstrained problem. The default number of variables is 30.
        .. note:: Continuous problem having a convex Pareto front
        """

        def __init__(self,number_of_objs):
            """ :param number_of_variables: Number of decision variables of the problem.
            """
            super(ZDT1_INT_TEST, self).__init__()
            self.number_of_variables = 6
            self.number_of_objectives = number_of_objs
            self.number_of_constraints = 1

            self.obj_directions = [self.MINIMIZE, self.MAXIMIZE, self.MINIMIZE][:number_of_objs]
            self.obj_labels = ['train_times', 'rewards', 'resource_wastages'][:number_of_objs]

            self.lower_bound = [1,1,1,1000,1,1] # [1,1,1,1000,1,1] #[10,5,50,10000,5,10]
            self.upper_bound = [100,100,500,50000,200,100]  #[100,100,500,50000,200,50] # [100,100,100,50000,10,30]#TODO Check later #ramake max iter to 100

        def evaluate(self, solution: IntegerSolution) -> IntegerSolution:

            solution.objectives[0] = calculate_objectives_score("train_times",corr_arr_train_times,f_list_train_times, 
                                                                solution.variables[0],
                                                                solution.variables[1],
                                                                solution.variables[2],
                                                                solution.variables[3],
                                                                solution.variables[4]/1000, #learning rate#
                                                                solution.variables[5] )
            solution.objectives[1] =  - calculate_objectives_score("rewards",corr_arr_rewards,f_list_rewards,
                                                                solution.variables[0],
                                                                solution.variables[1],
                                                                solution.variables[2],
                                                                solution.variables[3],
                                                                solution.variables[4]/1000, #learning rate
                                                                solution.variables[5] )
            if self.number_of_objectives > 2:
                solution.objectives[2] = calculate_objectives_score("resource_wastages",corr_arr_resource_wastages,f_list_resource_wastages,
                                                                    solution.variables[0],
                                                                    solution.variables[1],
                                                                    solution.variables[2],
                                                                    solution.variables[3],
                                                                    solution.variables[4]/1000, #learning rate
                                                                    solution.variables[5] )

            self.__evaluate_constraints(solution)

            return solution

        def __evaluate_constraints(self, solution: IntegerSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]

            d = solution.variables[3]
            b = solution.variables[2]
            i = solution.variables[1]
            ##MAX ITERATION NUMBER = DATASET_SIZE (d) / ( NUMBER OF DEVICES (K) * BATCH_SIZE (B) ) i.e 5000 / (5 * 100) = 10 Due to implementation issues with trainloader
            ##therefore i must be <= than max_iter_number
            max_iter_number = d / ( 5 * b )
            constraints[0] = max_iter_number - i

            solution.constraints = constraints

        
        def get_name(self):
            return 'ZDT1_TEST_INT'

    ##MOO RUNs
    for obj_nr in range_obj:
        for max_evals in range_evals:
            for pop_size in range_pop_size:
                if max_evals * pop_size < 10000000:
                    for mutation_prob in range_mutation_p:
                        for mutation_distr_i in range_mutation_dist_i:
                            for crossover_prob in range_crossover_p:
                                for crossover_distr_i in range_crossover_dist_i:
                                    for rel_idx in range(reliability_runs):
                                        offspring_size = pop_size

                                        run_counter += 1
                                        if run_counter % 10 == 1:
                                            print("## Progress: "+ str(run_counter) + "/" + str(total_runs))                                    

                                        problem = ZDT1_INT_TEST(obj_nr)

                                        algorithm = NSGAII(
                                            problem=problem,
                                            population_size=pop_size,
                                            offspring_population_size=offspring_size,
                                            mutation=IntegerPolynomialMutation(probability=mutation_prob/ problem.number_of_variables, distribution_index=mutation_distr_i),
                                            crossover=IntegerSBXCrossover(probability=crossover_prob, distribution_index=crossover_distr_i),
                                            termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
                                        )
                                        algorithm_name = "NSGAII"

                                        # algorithm = SMPSO(
                                        # problem=problem,
                                        # swarm_size=pop_size,
                                        # mutation=IntegerPolynomialMutation(probability=mutation_prob / problem.number_of_variables, distribution_index=mutation_distr_i),
                                        # leaders=CrowdingDistanceArchive(1000), #try bigger crowding distance
                                        # termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
                                        # )
                                        # algorithm_name = "SMPSO"

                                        label_name = algorithm_name +" " + str(obj_nr)+"d "+ str(max_evals)+"e "+str(pop_size)+"p "+str(mutation_prob) +"mp "+str(mutation_distr_i) +"md "+str(crossover_prob) +"cp "+str(crossover_distr_i) + "cd"

                                        print("### Alg running with params: " + label_name)
                                        algorithm.run()
                                        front = get_non_dominated_solutions(algorithm.get_result())

                                        hyper_volume = HyperVolume([10.0, 10.0, 10000.0]).compute([front[i].objectives for i in range(len(front))])

                                        #inverse obj for rewards
                                        for idx, item in enumerate(front):
                                            item.objectives[1] = - item.objectives[1] # Rewards are inversed in computation because NSGA2 can only minimize, not maximize, revert them back
                                            front[idx] = item

                                        #Sort the front
                                        if obj_nr == 2:
                                            front.sort(key= lambda x: (x.objectives[0],x.objectives[1]))
                                        elif obj_nr == 3:
                                            front.sort(key= lambda x: (x.objectives[0],x.objectives[1],x.objectives[2]))
                                        
                                        front,avg_crowd_d = set_crowding_distance(front)
                                        # Save results to file
                                        # print_function_values_to_file(front, "FUN" + label_name)
                                        # print_variables_to_file(front, "VAR" + label_name)

                                        #print(f"Algorithm: {algorithm.get_name()}")
                                        #print(f"Problem: {problem.get_name()}")
                                        print(f"Computing time: {round(algorithm.total_computing_time,3)}")

                                        from jmetal.lab.visualization import Plot

                                        # plot_front = Plot(title='Pareto front approximation', axis_labels=['Train time', 'Rewards' , 'Resource Wastage'])
                                        # plot_front.plot(front, label=label_name, filename=label_name + " R"+ str(rel_idx+4)+"_", format='png')

                                        ############CONCAT THE DF

                                        current_run_df = pd.DataFrame({'algorithm': [algorithm_name],
                                                                        'objective_number': [obj_nr],
                                                                        'evaluations': [max_evals],
                                                                        'population': [pop_size] ,
                                                                        'mutation_probability': [mutation_prob],
                                                                        'mutation_dist_idx' : [mutation_distr_i],
                                                                        'crossover_probability': [crossover_prob],
                                                                        'crossover_dist_idx' : [crossover_distr_i],
                                                                        'hypervolume' : [round(hyper_volume,5)],
                                                                        'avg_crowding_distance' : [round(avg_crowd_d,5)]
                                                                        })
                                        
                                        moo_df = pd.concat([moo_df,current_run_df],ignore_index= True)


                                        ####################save front variables
                                        list_variables = []
                                        counter_skip = 0
                                        for sol in front:
                                            max_iter_number =  sol.variables[3]/ ( 5 * sol.variables[2] )
                                            if sol.variables[1] <= max_iter_number: #So that the training cna run; minimum batch/dataset/iter size
                                                counter_skip += 1
                                                if counter_skip % 5 == 0:
                                                    counter_skip = 0
                                                    list_variables.append(sol.variables)



    print(list_variables)
    
    moo_df.to_pickle(file_path)
    print(moo_df)
    print("FINISH WITH JMETAL")

def set_crowding_distance(front):
    sum_crowding_d = 0
    for sol_index in range(1,len(front)-1):
        front[sol_index].crowding_distance = 0

        for m in range(front[sol_index].number_of_objectives):
            front[sol_index].crowding_distance = front[sol_index].crowding_distance + (front[sol_index+1].objectives[m] - front[sol_index-1].objectives[m])
            #TODO make multiplication to get AREA like in dragi's course OR make of addition like in paper? 
            # http://gpbib.cs.ucl.ac.uk/gecco2005/docs/p257.pdf PAGE 3
            # https://moodle.aau.at/pluginfile.php/999808/mod_resource/content/1/Lecture%208%20-%20IoT%20Multi%20Objective%20Application%20Scheduling%20in%20Cloud%20and%20Edge.pdf
            # Slide 18

        sum_crowding_d += front[sol_index].crowding_distance

    front[0].crowding_distance = sys.maxsize
    front[len(front)-1].crowding_distance = sys.maxsize

    return front , sum_crowding_d / len(front)
    #tread edge cases S[0], S[n]

def combine_duplicate_rows_of_dataframe(moo_df):
    moo_df = moo_df.groupby(['algorithm','objective_number','evaluations','population','mutation_probability','mutation_dist_idx','crossover_probability','crossover_dist_idx'],as_index=False).mean()
    moo_df = moo_df.sort_values(by=['hypervolume'], ascending = False)
    return moo_df

def visualize_boxplots(moo_df,name_of_value_for_boxplots,list_of_column_names):

    moo_df = combine_duplicate_rows_of_dataframe(moo_df)
    for c_name in list_of_column_names:
        visualize_boxplots_of_objective_based_on_variable(moo_df,name_of_value_for_boxplots,c_name)
        print("## Visualizing: " + c_name)

######################################
if __name__ == "__main__":
    
    run_MAIN_moo()

    # algorithm_name = "NSGAII"
    # file_path = config.home + './results/df_MOO_'+ algorithm_name + '_3d6.pkl'
    # moo_df = get_df_from_file(file_path)
    # # file_path = config.home + './results/df_MOO_'+ algorithm_name + '_3d1.pkl'
    # # moo_df2 = get_df_from_file(file_path)

    # # moo_df = pd.concat([moo_df,moo_df2],ignore_index= True)
    # print(moo_df)   
    # visualize_boxplots(moo_df,'hypervolume',['evaluations','population','mutation_probability','mutation_dist_idx','crossover_probability','crossover_dist_idx'])
    
    
    print("FINISH")



