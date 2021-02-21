# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:55:19 2021

@author: user
"""

#Correction for Boston Dataset - taken from a point in Ashland, and from a point on Nahant

import pandas as pd
df = pd.read_csv("boston_corrected.csv")

ini = df.iloc[0] #NAHANT
ash = df[df['CMEDV'] == 21.9][df['TOWN'] =='Ashland']

#Coefficient corrections

def correction(latcorr1, loncorr1, latcorr2, loncorr2, ini1, ini2):
    a = (loncorr1-loncorr2)/(ini1['LON'].item() - ini2['LON'].item())
    b = loncorr1 - a*ini1['LON'].item()
    c = (latcorr1-latcorr2)/(ini1['LAT'].item() - ini2['LAT'].item())
    d = latcorr1 - c*ini1['LAT'].item()
    
    
    return a, b, c, d

a, b, c, d = correction(42.426, -70.9278, 42.259635, -71.465395, ini, ash)
print(a, b, c, d)

df["LON"] = a*df["LON"] + b
df["LAT"] = c*df["LAT"] + d