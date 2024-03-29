import numpy as np
import matplotlib.pyplot as mlp
import pandas as pd

# Importing and prerocessing dataset
dataset = pd.read_csv('D:\Machine_Learning_course\Machine Learning A-Z Template Folder\Part 5 - Association Rule Learning\Section 28 - Apriori\Python\Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j])for j in range(0, 20)])
    
# Using apriori 
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Inspecting rules
result = list(rules)
print(result)

# Fuction for inspecting the resultant from the aprior model - not important to have it in mind
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(result), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Sorting the resultset by lift
sortedResultSet = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')