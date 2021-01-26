# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split; "kaynak = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
from sklearn import metrics
from sklearn.metrics import accuracy_score

"reading lines from file, kaynak = https://www.youtube.com/watch?v=efpFDaXOG6Y"
dataText = open("./spambaseData.txt","r");
dataLines = dataText.readlines(); "txt deki her satırı listeye aktar"
dataText.close()

"2d liste oluşturma = https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/ "
dataSet = [[0]*58]*len(dataLines)

for i in range(len(dataLines)):
    dataSet[i] = dataLines[i].split(","); "',' gördüğün yerde böl = https://mkyong.com/python/python-how-to-split-a-string/"  

for i in range(len(dataLines)):
    for s in range(58):
        dataSet[i][s] = float(dataSet[i][s])

"normail listeden matris liste oluştur,kaynak = https://stackoverflow.com/questions/29224148/how-to-construct-a-matrix-from-lists-in-python"
dataSetMatris = np.array(dataSet)

data = dataSetMatris [:,0:48]
data2 = dataSetMatris [:,0:54]
spam = dataSetMatris [:,-1]

ogreticiData, testData, ogreticiSpam, testSpam = train_test_split(data,spam, test_size=.20)

"Naive bayes öğretme ve test kaynak = https://www.youtube.com/watch?v=99MN-rl8jGY&list=PLmAOJTFW26iRWJJ4y_uLUKOiXCE9hJ2a1&index=5&t=0s"
naiveMulti = MultinomialNB()
naiveMulti.fit(ogreticiData,ogreticiSpam)
spamTahmin = naiveMulti.predict(testData);    "test için ayırılan kısmı tahmin et"
print("Multinomial, Sadece Kelimeler İle Verimlilik:")
print (accuracy_score(testSpam,spamTahmin)) 

naiveGauss = GaussianNB()
naiveGauss.fit(ogreticiData,ogreticiSpam)
spamTahmin = naiveGauss.predict(testData)
print("Gaussian, Sadece Kelimeler İle Verimlilik:")
print(accuracy_score(testSpam,spamTahmin))

naiveBernoulli = BernoulliNB(binarize = 0.1)
naiveBernoulli.fit(ogreticiData,ogreticiSpam)
spamTahmin = naiveBernoulli.predict(testData)
print("Bernoulli, Sadece Kelimeler İle Verimlilik:")
print (accuracy_score(testSpam,spamTahmin))

