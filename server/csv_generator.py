import os
import pandas as pd
import numpy as np

results = []

estimations = pd.read_csv("smooth_estimates.csv")
estimations = estimations.dropna()

for filename in os.listdir('processed_data'):
    if filename.endswith(".npy"):
        name = filename.replace(".npy", "")
        