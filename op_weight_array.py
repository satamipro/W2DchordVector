import numpy as np;
import pandas as pd;

def getchordvector(weight, dictionary):
    for value in dictionary.keys():
        dictionary[value] = np.dot(dictionary[value], weight)
    return dictionary

def getcosinesimilarity(dictionary):
    rownames = list(dictionary.keys())
    colnames = rownames
    print(colnames)
    tmp = np.zeros((len(dictionary), len(dictionary)))
    tmp2 = pd.DataFrame(tmp, index = rownames, columns = colnames)
    cosinesimilarity = tmp2.copy()
    #print(cosinesimilarity.loc['Am', 'C'])
    for i in dictionary.keys():
        for j in dictionary.keys():
            cosinesimilarity.loc[i, j] = np.dot(dictionary[i], dictionary[j]) / (np.linalg.norm(dictionary[i]) * np.linalg.norm(dictionary[j]))
    return cosinesimilarity