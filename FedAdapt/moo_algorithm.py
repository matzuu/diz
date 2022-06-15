



def initial_NSGAII_test():
    from jmetal.algorithm.multiobjective import NSGAII
    from jmetal.operator import SBXCrossover, PolynomialMutation
    from jmetal.problem import ZDT1
    from jmetal.util.termination_criterion import StoppingByEvaluations

    problem = ZDT1()

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    algorithm.run()

    #########################################################

    from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file

    front = get_non_dominated_solutions(algorithm.get_result())

    # save to files
    print_function_values_to_file(front, 'FUN.NSGAII.ZDT1')
    print_variables_to_file(front, 'VAR.NSGAII.ZDT1')


    #############################

    from jmetal.lab.visualization import Plot

    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    plot_front.plot(front, label='NSGAII-ZDT1', filename='NSGAII-ZDT1', format='png')

#########################################################################################################
#########################################################################################################
#########################################################################################################

def subsetSum_NSGAII_example():
    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.operator import BitFlipMutation, SPXCrossover
    from jmetal.problem.multiobjective.unconstrained import SubsetSum
    from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
    from jmetal.util.termination_criterion import StoppingByEvaluations

    C = 300500
    W = [
        2902,
        5235,
        357,
        6058,
        4846,
        8280,
        1295,
        181,
        3264,
        7285,
        8806,
        2344,
        9203,
        6806,
        1511,
        2172,
        843,
        4697,
        3348,
        1866,
        5800,
        4094,
        2751,
        64,
        7181,
        9167,
        5579,
        9461,
        3393,
        4602,
        1796,
        8174,
        1691,
        8854,
        5902,
        4864,
        5488,
        1129,
        1111,
        7597,
        5406,
        2134,
        7280,
        6465,
        4084,
        8564,
        2593,
        9954,
        4731,
        1347,
        8984,
        5057,
        3429,
        7635,
        1323,
        1146,
        5192,
        6547,
        343,
        7584,
        3765,
        8660,
        9318,
        5098,
        5185,
        9253,
        4495,
        892,
        5080,
        5297,
        9275,
        7515,
        9729,
        6200,
        2138,
        5480,
        860,
        8295,
        8327,
        9629,
        4212,
        3087,
        5276,
        9250,
        1835,
        9241,
        1790,
        1947,
        8146,
        8328,
        973,
        1255,
        9733,
        4314,
        6912,
        8007,
        8911,
        6802,
        5102,
        5451,
        1026,
        8029,
        6628,
        8121,
        5509,
        3603,
        6094,
        4447,
        683,
        6996,
        3304,
        3130,
        2314,
        7788,
        8689,
        3253,
        5920,
        3660,
        2489,
        8153,
        2822,
        6132,
        7684,
        3032,
        9949,
        59,
        6669,
        6334,
    ]

    problem = SubsetSum(C, W)

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(probability=0.5),
        crossover=SPXCrossover(probability=0.8),
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################



    


def problem_mixedIntFloat_TEST():

    from jmetal.core.problem import FloatProblem, BinaryProblem, Problem, IntegerProblem
    from jmetal.core.solution import FloatSolution, BinarySolution, CompositeSolution, IntegerSolution
    import random
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
            self.number_of_constraints = 0

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

            return solution

        def get_name(self):
            return 'ZDT1_TEST_INT'

    for obj_nr in [2,3]:
        for max_evals in [100,25000]:

            #obj_nr = 2            
            #max_evals = 25000
            pop_size = 100
            offspring_size = pop_size
            mutation_prob = 1.0 #[0.0,1.0]
            mutation_distr_i = 20 #[5.0, 400.0]
            crossover_prob = 1.0 #[0.0, 1.0]
            crossover_distr_i = 20 #[5.0, 400.0]
           



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
            # swarm_size=100,
            # mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            # leaders=CrowdingDistanceArchive(100), #try bigger crowding distance
            # termination_criterion=StoppingByEvaluations(max_evaluations=max_evals)
            # )
            # algorithm_name = "SMPSO"

            label_name = algorithm_name + str(obj_nr)+"d_"+ str(max_evals)+"e_"+str(pop_size)+"p_"+str(mutation_prob) +"mp_"+str(mutation_distr_i) +"md_"+str(crossover_prob) +"cp_"+str(crossover_distr_i) + "cd"

            print("ALG RUNNING WITH PARAMS: " + label_name)
            algorithm.run()
            front = get_non_dominated_solutions(algorithm.get_result())

            # Save results to file
            #print_function_values_to_file(front, "FUN." + algorithm.label)
            #print_variables_to_file(front, "VAR." + algorithm.label)

            print(f"Algorithm: {algorithm.get_name()}")
            print(f"Problem: {problem.get_name()}")
            print(f"Computing time: {algorithm.total_computing_time}")

            from jmetal.lab.visualization import Plot

            plot_front = Plot(title='Pareto front approximation', axis_labels=['train_times', 'rewards' , 'resource_wastage'])
            plot_front.plot(front, label=label_name, filename=label_name, format='png')

    print("FINISH WITH JMETAL")


######################################
if __name__ == "__main__":
    #initial_NSGAII_test()
    # subsetSum_NSGAII_example()
    # problem_first_implementation_attempt()
    problem_mixedIntFloat_TEST()
    print("FINISH")



