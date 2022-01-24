import pickle #
with open('./results/RL_Metrics1.pkl', 'rb') as f:
    datares1 = pickle.load(f)

#METRICS DICT -> EPISODES_DICT -> STEPS DICT -> small metrics
print("DONE")