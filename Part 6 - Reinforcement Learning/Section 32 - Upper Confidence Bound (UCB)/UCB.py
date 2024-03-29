import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 6 - Reinforcement Learning\Section 32 - Upper Confidence Bound (UCB)\Python\Ads_CTR_Optimisation.csv')

# Using UCB
import math
N = 10000
d = 10 # Each round
ads_selected = []
number_of_selections = [0] * d  
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(number_of_selections[i]) > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]