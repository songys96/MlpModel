import pandas as pd
import numpy as np

idx = [10,20,30,40,50]
name = ['s','p','k','c','g']

df = pd.DataFrame({"id": idx, 'name':name})
arr = df.iloc[:,0].to_numpy()
for i in arr:
    print(i)