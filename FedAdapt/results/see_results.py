import pickle
with open('FedAdapt_res.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)

with open('Client_time_metrics.pkl', 'rb') as f2:
    data2 = pickle.load(f2)

print(data2)