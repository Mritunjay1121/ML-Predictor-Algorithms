import sklearn
import pandas as pd
import math, datetime
import numpy as np
from math import sqrt
from sklearn import preprocessing, model_selection, svm
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import random


accuracies=[]
for i in range(25):
    df=pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data=df.astype(float).values.tolist() # To convert all data into float
    random.shuffle(full_data)
    # print(df)



    def k_neartest_neighbors(data,predict,k=3):
        if len(data)>=k:
            warnings.warn('K is set to a value less than the total groups')
        distances=[]
        for group in data:
            for features in data[group]:
                euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
                distances.append([euclidean_distance,group])
    #     print(sorted(distances))
        votes=[i[1] for i in sorted(distances)[:k]]
    #     print(votes)
    #     print("nhjh",Counter(votes).most_common(1))
        vote_result=Counter(votes).most_common(1)[0][0]
        confidence=Counter(votes).most_common(1)[0][1]/k
        return vote_result,confidence # confidence = vote_result/votes....it is the probability of the output



    # Classifying the data into Train and Test set

    test_size=0.4
    train_set={2:[],4:[]}
    test_set={2:[],4:[]}
    train_data=full_data[:-int(test_size*len(full_data))]
    test_data=full_data[-int(test_size*len(full_data)):]



    # Making train and test set with classified labels

    for a in train_data:
        train_set[a[-1]].append(a[:-1])

    for b in test_data:
        test_set[b[-1]].append(b[:-1])

    # style.use('fivethirtyeight')

    correct=0
    total=0
    for classes in test_set:
        list_ofa_class=test_set[classes]
        for points in list_ofa_class:
            vote,confidence= k_neartest_neighbors(train_set,points,k=200)
            if vote==classes:
                correct+=1
            else:
                confidence=confidence*confidence
                #print(confidence)
            total+=1
#     print("Accuracy : ",correct/total,"Total confidence: ",confidence)
    accuracies.append(correct/total)
print(sum(accuracies)/len(accuracies))








# result=k_neartest_neighbors(dataset,new_features,k=3)
# print(result)

