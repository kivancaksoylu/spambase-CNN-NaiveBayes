# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:24:43 2020

@author: aksoy
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split; "kaynak = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense , Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import pandas as pd
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

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
spam = dataSetMatris [:,-1]

texts_ = pad_sequences(data,maxlen=48)

ogreticiData, testData, ogreticiSpam, testSpam = train_test_split(data,spam, test_size=0.70)

max_futures=200
maxlen=48
batch_size=40 
embedding_dims=300; 
kernel_size=48; "Conv1D de öğrenim için kullanılan datanın seçileceği pencere büyüklüğü bu durumda listeden 48x3 pencere boyutu"
hidden_dims=300; "gizli katman nöronları" 
epochs=30; "kaç kere eğitime sokulacak"
filters=250; "çıkşta oluşturulacak filtre sayısı"

model= Sequential(); "art arda layerlar"
model.add(Embedding(max_futures, embedding_dims,input_length=maxlen) ); "Embedding katmanının fonksiyonu"
"max_futures = toplam farklı değer sayısı, embedding_dims= embedding nöronları ve dönüştürülen vektörün uzunluğu, input_length= girdi uzunluğu."

model.add(Dropout(0.2)); "eğitimdeki bağzı nöronlar bağlarıyla birlikte iptal edilir, bundaki amaç overfitting probleminin üstesinden gelmektir"
"overfitting algoritmanın kullandığı veri setine çok fazla uyum sağlaması ve başka veri setleriyle çalışamamasıdır."

model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)); "Conv1D'de liste aşağı doğru taranır, 1 satır 1 boyut olarak görünür"
"padding=valid çıkışta oluşacak bilginin(matris) giriştekinden küçük olması"
"strides= kernel_size ile oluşturulan pencerenin aşağı doğru kaçar kaçar atlayarak tarayacağı"
"activation relu=relu fonksiyonu aldığı değerleri 0 dan bük olarak olduğu gibi gönderir, 0 dan küçük ise 0 olarak gönderir."

model.add(GlobalMaxPooling1D()); "1D pooling"

model.add(Dense(hidden_dims)); "Dense= girilen değer çıkış nöronlarının sayısını belirler"
"bu layerde giriş nöronları ile çıkış nöronları arasında weight matrisi bulunur bu matrisin boyutu giriş ve çıkış nöron sayısına göre otomatik hesaplanır"

model.add(Dropout(0.2))

model.add(Activation('relu')); "hidden dims nöronlarının hangi fonksiyonu kullanacağı"
model.add(Dense(1)); "çıkış katmanı"

model.add(Activation('sigmoid')); "sigmoid aldığı değerleri 1 veya 0 a dönüştürür"

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"compile= eklenen bütün katmanları ve mimariyi derler"


t = model.fit(ogreticiData, ogreticiSpam, batch_size=batch_size, epochs=epochs, validation_data=(testData, testSpam)); 
"fit fonksiyonu training in yapıldığı bölüm"


la_ratio=model.evaluate(testData,testSpam)
print('Loss/Accuracy :', la_ratio)


plt.plot(t.history['accuracy'],color='b', label="Training accuracy")
plt.plot(t.history['val_accuracy'],color='y', label="Test accuracy")
plt.plot(t.history['loss'],color='g', label="Training loss")
plt.plot(t.history['val_loss'],color='r', label="Test accuracy")
plt.title('Model')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy','Test Accuracy','Train Loss','Test Loss'],loc='bottom left')
print(plt.show())

# rounded_pred = model.predict_classes(testData.reshape(-1,48), batch_size=128, verbose=0)
# rounded_labels=np.argmax(testSpam.reshape(-1,1), axis=1); "reshape = https://stackoverflow.com/questions/35401041/concatenation-of-2-1d-numpy-arrays-along-2nd-axis"
# print(confusion_matrix(rounded_labels,rounded_pred))
# print(plot_confusion_matrix(conf_mat=confusion_matrix(rounded_labels,rounded_pred)))


