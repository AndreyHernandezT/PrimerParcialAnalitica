#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
file = 'AirQualityClean.csv'
dataset = pd.read_csv(file, sep=';')

dataset.drop(['CO(GT)','Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)

dataset.replace(to_replace=',',value='.',regex=True,inplace=True)
for i in 'C6H6(GT) T RH AH'.split():
    dataset[i] = pd.to_numeric(dataset[i],errors='coerce')
    
dataset.replace(to_replace=-200,value=np.nan,inplace=True)

dataset['Date'] = pd.to_datetime(dataset['Date'],dayfirst=True) 
dataset['Time'] = pd.to_datetime(dataset['Time'],format= '%H.%M.%S' ).dt.time

dataset.drop('NMHC(GT)', axis=1, inplace=True) 

qulo1 = dataset.quantile(0.25) 
qulo3 = dataset.quantile(0.75) 
IQR = qulo3 - qulo1 
scale = 2
wuebos = qulo1 - scale*IQR
Tetas = qulo3 + scale*IQR
lower_Shet = (dataset[dataset.columns[2:13]] < wuebos)
Pancha = (dataset[dataset.columns[2:13]] > Tetas)

num_cols = list(dataset.columns[2:13])
PanochaIQR = dataset[~((dataset[num_cols] < (qulo1 - scale * IQR)) |(dataset[num_cols] > (qulo3 + scale * IQR))).any(axis=1)]

pd.options.mode.chained_assignment = None
PanochaIQR.drop(['NOx(GT)','NO2(GT)'],axis=1, inplace=True)

dataset_tits = PanochaIQR.dropna(how='any', axis=0)
dataset_tits.reset_index(drop=True,inplace=True)

dataset_tits['Week Day'] = dataset_tits['Date'].dt.day_name()

cols = dataset_tits.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:11]
dataset_tits = dataset_tits[cols]

dataset_wed = dataset_tits[dataset_tits['Week Day'] == 'Wednesday']

dataset_tits.drop('C6H6(GT)', axis=1, inplace=True)
Y = dataset_tits['PT08.S1(CO)'] 
X = dataset_tits.drop(['PT08.S1(CO)','Date', 'Time', 'Week Day'], axis=1)


for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

    modelo = LinearRegression()
    
    modelo.fit(X = np.array(X_train), y = Y_train)
    pred_modelo = modelo.predict(X_test)#Concentraciones de CO pronosticadas

    print("Iteración: ",i)
    print('\tModelo de regresión: R²={:.2f}'.format(metrics.r2_score(Y_test, pred_modelo)))