# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pywt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure

mydata = pd.read_csv("FinalData.csv")

lst = mydata.iloc[0].to_numpy()
lst = lst[1:]

mydata = mydata.T # transpose of the data frame

ind = mydata.index.values.tolist()
ind = ind[1:]

# print (ind)
# print (lst)

# %%

(cA, cD) = pywt.dwt(lst, 'db2', 'smooth') # using db2 wavelet function to decompose data 
A = pywt.idwt(cA, None, 'db2', 'smooth') # using inverse wavelet to reconstruct linear components
D = pywt.idwt(None, cD, 'db2', 'smooth') # using inverse wavelet to reconstruct non-linear components

# lst_rec = A + D # reconstruction of lst i.e data before decomposition


# %%
d = {'data': lst, 'linear components': A, 'non-linear components': D} # constructing map with non-decomposed and decomposed data
df = pd.DataFrame(d) # map -> data frame
df.index = ind 

print (df)

# figure(num=None, figsize=(100, 10), dpi=80, facecolor='w', edgecolor='k')
pl = df.plot(figsize=(100, 10), grid=True)

mpl.style.use('seaborn')
pl.set_title('Decomposition of data into linear and non-linear components'.format('seaborn'), color='C0')

pl.legend()

plt.show()
