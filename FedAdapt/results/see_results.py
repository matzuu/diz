import pickle
with open('./results/FedAdapt_res1OldRl.pkl', 'rb') as f:
    datares1 = pickle.load(f)

with open('./results/Client_time_metrics1OldRl.pkl', 'rb') as f2:
    datametrics1 = pickle.load(f2)

with open('./results/Client_time_metrics1NoOFF.pkl', 'rb') as f3:
    datametrics1Off = pickle.load(f3)

with open('./results/Client_time_metrics2.pkl', 'rb') as f4:
    datametrics2 = pickle.load(f4)

with open('./results/Client_time_metrics3.pkl', 'rb') as f5:
    datametrics3 = pickle.load(f5)

with open('./results/Client_time_metrics4.pkl', 'rb') as f6:
    datametrics4 = pickle.load(f6)

print(datametrics2)