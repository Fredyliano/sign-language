import pickle

with open('modelSVM.p', 'rb') as f:
    data = pickle.load(f)

print(data)