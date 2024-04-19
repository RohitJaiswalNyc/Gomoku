import numpy as np
import torch as tr
import itertools as it
import matplotlib.pyplot as pt
import random
from scipy.signal import correlate
import pickle

def save():
    with open('listfile', 'ab') as fp:
        pickle.dump(1, fp)
        pickle.dump(2, fp)
        pickle.dump(1, fp)


data = []
with open('listfile', 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass

save()
save()
print(data)