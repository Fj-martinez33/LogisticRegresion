#from utils import db_connect
#engine = db_connect()

# Librerias

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np
import json
from pickle import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

#Recopilamos datos
data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=";")

#Además, vamos a guardarlo en el repositorio por seguridad.
raw_data_backup = data.to_csv("../data/raw/raw_data.csv", sep=";")

#Obtenemos dimensiones del dataset
data.shape

#Obtenemos informacion sobre los tipos de datos.
data.info()

#Obtenemos info sobre valores nulos

data.isna().sum().sort_values(ascending=False)

#Funcion para eliminar duplicados

#Columna identificadora del Dataset.
#id = "id" En este caso no es necesario porque no hay un identificador en el dataset

def EraseDuplicates(dataset):
    older_shape = dataset.shape
    if (dataset.duplicated().sum()):
        print ("Erase duplicates...")
        dataset.drop_duplicates(inplace = True)
        print(f"Total number of duplicates {dataset.duplicated().sum()}")
    else:
        print ("No coincidences.")
        pass
    
    print (f"The older dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    
    return dataset

EraseDuplicates(data)

#Funcion para eliminar datos irrelevantes.

irrelevant_lst = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

def EraseIrrelevants(dataset, lst):
    older_shape = data.shape
    print("Erase irrelevant´s dates...")
    dataset.drop(lst, axis = 1, inplace = True)
    print (f"The old dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    return dataset

EraseIrrelevants(data, irrelevant_lst)

# Analisis sobre variables categoricas

def CategoricGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(4, 3, figsize=(20,15))

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0,0], data = dataset, x = "job")
    sns.histplot( ax = axis[0,1], data = dataset, x = "marital").set(ylabel = None)
    sns.histplot( ax = axis[0,2], data = dataset, x = "education").set(ylabel = None)
    sns.histplot( ax = axis[1,0], data = dataset, x = "default")
    sns.histplot( ax = axis[1,1], data = dataset, x = "housing").set(ylabel = None)
    sns.histplot( ax = axis[1,2], data = dataset, x = "loan").set(ylabel = None)
    sns.histplot( ax = axis[2,0], data = dataset, x = "contact")
    sns.histplot( ax = axis[2,1], data = dataset, x = "month").set(ylabel = None)
    sns.histplot( ax = axis[2,2], data = dataset, x = "day_of_week").set(ylabel = None)
    sns.histplot( ax = axis[3,0], data = dataset, x = "pdays")
    sns.histplot( ax = axis[3,1], data = dataset, x = "poutcome").set(ylabel = None)
    sns.histplot( ax = axis[3,2], data = dataset, x = "previous").set(ylabel = None)
   

    #Mostramos el grafico.
    plt.tight_layout()
    plt.show()

CategoricGraf(data)

print (data["job"].value_counts())
print (data["education"].value_counts())

# Analisis sobre variables numericas

def NumericalGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(2, 3, figsize=(10,5), gridspec_kw={"height_ratios" : [6,1]})

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0,0], data = dataset, x = "age", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[1,0], data = dataset, x = "age")
    sns.histplot( ax = axis[0,1], data = dataset, x = "duration", kde=True).set(xlabel=None, ylabel = None)
    sns.boxplot( ax = axis[1,1], data = dataset, x = "duration")
    sns.histplot( ax = axis[0,2], data = dataset, x = "campaign", kde=True).set(xlabel = None, ylabel = None)
    sns.boxplot( ax = axis[1,2], data = dataset, x = "campaign")
    
    plt.tight_layout()
    plt.show()

NumericalGraf(data)

#Analisis numerico/numerico

def NumNumAnalysi(dataset, x, y_list):
    #Creamos la figura
    fig, axis = plt.subplots(2, 1, figsize=(10,8))

    #Creamos la grafica
    sns.regplot( ax = axis[0], data = dataset, x = y_list[0], y = x)
    sns.heatmap( data[[x,y_list[0]]].corr(), annot=True, fmt=".2f", ax = axis[1], cbar=False)

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()

NumNumAnalysi(data, "campaign", ["duration"])

#Analisis categorico/categorico

def CatCatAnalysi(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(4, 3, figsize=(20,15))

    #Creamos las graficas.
    sns.countplot(ax = axis[0,0], data = dataset, x = "job", hue = "y")
    sns.countplot(ax = axis[0,1], data = dataset, x = "marital", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[0,2], data = dataset, x = "education", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[1,0], data = dataset, x = "default", hue = "y")
    sns.countplot(ax = axis[1,1], data = dataset, x = "housing", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[1,2], data = dataset, x = "loan", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[2,0], data = dataset, x = "contact", hue = "y")
    sns.countplot(ax = axis[2,1], data = dataset, x = "month", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[2,2], data = dataset, x = "day_of_week", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[3,0], data = dataset, x = "pdays", hue = "y")
    sns.countplot(ax = axis[3,1], data = dataset, x = "poutcome", hue = "y").set(ylabel = None)
    sns.countplot(ax = axis[3,2], data = dataset, x = "previous", hue = "y").set(ylabel = None)

    plt.tight_layout()
    plt.show()

CatCatAnalysi(data)

#Vamos a factorizar las variables categoricas -esta vez sin OHE.-

def Factorized(dataset, col):

    factorize = pd.factorize(dataset[col])
    parsin_dic = {}
    index = factorize[1]
    factor = list(set(factorize[0]))
    for i in range (len(factorize[1])):
        parsin_dic.update({index[i] : int(factor[i])})
    
    with open (f"../data/interim/{col}_parsing.json", "w") as j:
        json.dump(parsin_dic, j)
    
    dataset[col] = factorize[0]
    
    return dataset

Factorized(data, "job")
Factorized(data, "marital")
Factorized(data, "education")
Factorized(data, "default")
Factorized(data, "housing")
Factorized(data, "loan")
Factorized(data, "contact")
Factorized(data, "month")
Factorized(data, "day_of_week")
Factorized(data, "poutcome")
Factorized(data, "y")

#Tabla de correlaciones
fig, axis = plt.subplots(figsize=(10,7))

sns.heatmap(data[["y", "age","job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week","duration","campaign", "pdays", "poutcome"]].corr(), annot=True, fmt=".2f")

plt.tight_layout()
plt.show()

#Corroboración de la tabla
fig, axis = plt.subplots(1,3,figsize=(15,5))

sns.regplot(ax = axis[0], data = data, x = "duration", y = "y")
sns.regplot(ax = axis[1], data = data, x = "pdays", y = "y").set(ylabel = None)
sns.regplot(ax = axis[2], data = data, x = "poutcome", y = "y").set(ylabel = None)

plt.tight_layout()
plt.show()

sns.pairplot(data)

# Comprobamos las metricas de la tabla.

data.describe()

#Grafica de outliers

fig, axis = plt.subplots(1, 3, figsize=(10,5))

sns.boxplot( ax = axis [0], data = data, y = "age")
sns.boxplot( ax = axis [1], data = data, y = "duration")
sns.boxplot( ax = axis [2], data = data, y = "campaign")

plt.tight_layout()
plt.show()

#Hacemos dos copias del dataset, una para el dataset con outliers y otra sin.

data_with_outliers = data.copy()
data_without_outliers = data.copy()

#Creamos una funcion para transformar los outliers.

def TransOutliers(dataset, col_outliers):
    stats = dataset[col_outliers].describe()
    
    #Establecemos los límites.
    # Los valores óptimos para sumarle al Q3 suelen ser 1.5*IQR, 1.75*IQR y 2*IQR.
    iqr = stats["75%"] - stats["25%"]
    upper_limit = float(stats["75%"] + (2 * iqr))
    lower_limit = float(stats["25%"] - (2 * iqr))
    
    if (lower_limit < 0):
        lower_limit = 0

    #Ajustamos el outlier por encima.
    dataset[col_outliers] = dataset[col_outliers].apply(lambda x : upper_limit if (x > upper_limit) else x)

    #Ajustamos el outlier por debajo.
    dataset[col_outliers] = dataset[col_outliers].apply(lambda x : lower_limit if (x < lower_limit) else x)

    #Guardamos los límites en un json.

    with open (f"../data/interim/outerliers_{col_outliers}.json", "w") as j:
        json.dump({"upper_limit" : upper_limit, "lower_limit" : lower_limit}, j)

    return dataset

TransOutliers (data_without_outliers, "age")
TransOutliers (data_without_outliers, "duration")
TransOutliers (data_without_outliers, "pdays")

data_without_outliers

#Comprobamos si existen valores faltantes.

data_with_outliers.isna().sum().sort_values()

data_without_outliers.isna().sum().sort_values()

# Primero dividimos los dataframes entre test y train

features = ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration", "campaign", "pdays", "previous", "poutcome"]
target_feature = ["y"]

def SplitData (dataset, num_features, target):
    x = dataset.drop(target, axis = 1)[num_features]
    y = dataset[target].squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)

    return x_train, x_test, y_train, y_test

x_train_with_outliers, x_test_with_outliers, y_train, y_test = SplitData(data_with_outliers, features, target_feature)
x_train_without_outliers, x_test_without_outliers,_, _ = SplitData(data_without_outliers, features, target_feature)

y_train.to_csv("../data/processed/y_train.csv")
y_test.to_csv("../data/processed/y_test.csv")

#Tenemos que escalar los dataset con Normalizacion y con Escala mM (min-Max)

#Normalizacion
def StandardScaleData(dataset, num_features):
    scaler = StandardScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = num_features)
    
    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/standar_scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/standar_scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_standarscale = StandardScaleData(x_train_with_outliers, features)
x_train_without_outliers_standarscale = StandardScaleData(x_train_without_outliers,features)
x_test_with_outliers_standscale = StandardScaleData(x_test_with_outliers, features)
x_test_without_outliers_standscale = StandardScaleData(x_test_without_outliers, features)

#Escala mM
def MinMaxScaleData(dataset, num_features):
    scaler = MinMaxScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = num_features)

    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_mMScaler = MinMaxScaleData(x_train_with_outliers, features)
x_train_without_outliers_mMScaler = MinMaxScaleData(x_train_without_outliers,features)
x_test_with_outliers_mMScaler = MinMaxScaleData(x_test_with_outliers, features)
x_test_without_outliers_mMScaler = MinMaxScaleData(x_test_without_outliers, features)

#Seleccion de caracteristicas
k = 15
def SelectFeatures(dataset, y, filename, k = k):
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    return x_sel

#Dataset sin normalizacion
x_train_sel_with_outliers = SelectFeatures(x_train_with_outliers, y_train, "x_train_with_outliers")
x_test_sel_with_outliers = SelectFeatures(x_test_with_outliers, y_test, "x_test_with_outliers")
x_train_sel_without_outliers = SelectFeatures(x_train_without_outliers, y_train, "x_train_without_outliers")
x_test_sel_without_outliers = SelectFeatures(x_test_without_outliers, y_test, "x_test_without_outliers")

#Dataset Normalizado
x_train_sel_with_outliers_standarscale = SelectFeatures(x_train_with_outliers_standarscale, y_train, "x_train_with_outliers_standarscale")
x_test_sel_with_outliers_standarscale = SelectFeatures(x_test_with_outliers_standscale, y_test, "x_test_with_outliers_standscale")
x_train_sel_without_outliers_standarscale = SelectFeatures(x_train_without_outliers_standarscale, y_train, "x_train_sel_without_outliers_standarscale")
x_test_sel_without_outliers_standarscale = SelectFeatures(x_test_without_outliers_standscale, y_test, "x_test_without_outliers_standscale")

#Dataset Escalado min-Max
x_train_sel_with_outliers_mMScale = SelectFeatures(x_train_with_outliers_mMScaler, y_train, "x_test_with_outliers_mMScaler")
x_test_sel_with_outliers_mMScale = SelectFeatures(x_test_with_outliers_mMScaler, y_test, "x_test_with_outliers_mMScaler")
x_train_sel_without_outliers_mMScale = SelectFeatures(x_train_without_outliers_mMScaler, y_train, "x_train_without_outliers_mMScaler")
x_test_sel_without_outliers_mMScale = SelectFeatures(x_test_with_outliers_mMScaler, y_test, "x_test_with_outliers_mMScaler")

#Para acabara añadimos el target a los datasets.

target = "y"
def AgreeTarget(dataset, y, target = target):
    dataset[target] = list(y) #Si no le pones el list le pasa los indices y se crear valores NaN
    return dataset

AgreeTarget(x_train_sel_with_outliers, y_train)
AgreeTarget(x_test_sel_with_outliers, y_test)
AgreeTarget(x_train_sel_without_outliers, y_train)
AgreeTarget(x_test_sel_without_outliers, y_test)
AgreeTarget(x_train_sel_with_outliers_standarscale, y_train)
AgreeTarget(x_test_sel_with_outliers_standarscale, y_test)
AgreeTarget(x_train_sel_without_outliers_standarscale, y_train)
AgreeTarget(x_test_sel_without_outliers_standarscale, y_test)
AgreeTarget(x_train_sel_with_outliers_mMScale, y_train)
AgreeTarget(x_test_sel_with_outliers_mMScale, y_test)
AgreeTarget(x_train_sel_without_outliers_mMScale, y_train)
AgreeTarget(x_test_sel_without_outliers_mMScale, y_test)

#Para acabar nos guardamos los datasets en un csv

def DataToCsv(dataset, filename):
    return dataset.to_csv(f"../data/processed/{filename}.csv")

DataToCsv(x_train_sel_with_outliers, "x_train_sel_with_outliers")
DataToCsv(x_test_sel_with_outliers, "x_test_sel_with_outliers")
DataToCsv(x_train_sel_without_outliers, "x_train_sel_without_outliers")
DataToCsv(x_test_sel_without_outliers, "x_test_sel_without_outliers")
DataToCsv(x_train_sel_with_outliers_standarscale, "x_train_sel_with_outliers_standarscale")
DataToCsv(x_test_sel_with_outliers_standarscale, "x_test_sel_with_outliers_standarscale")
DataToCsv(x_train_sel_without_outliers_standarscale, "x_train_sel_without_outliers_standarscale")
DataToCsv(x_test_sel_without_outliers_standarscale, "x_test_sel_without_outliers_standarscale")
DataToCsv(x_train_sel_with_outliers_mMScale, "x_train_sel_with_outliers_mMScale")
DataToCsv(x_test_sel_with_outliers_mMScale, "x_test_sel_with_outliers_mMScale")
DataToCsv(x_train_sel_without_outliers_mMScale, "x_train_sel_without_outliers_mMScale")
DataToCsv(x_test_sel_without_outliers_mMScale, "x_test_sel_without_outliers_mMScale")

y_test

#MACHINE LEARNING

traindfs = [x_train_sel_with_outliers_standarscale, x_train_sel_without_outliers_standarscale, x_train_sel_with_outliers_mMScale, x_train_sel_without_outliers_mMScale]
testdfs = [ x_test_sel_with_outliers_standarscale, x_test_sel_without_outliers_standarscale, x_test_sel_with_outliers_mMScale, x_test_sel_without_outliers_mMScale]


def Training(traindataset, testdataset):
    results = []
    models = []

    for i in range(len(traindataset)):
        model = LogisticRegression()
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])
        result = {"index" : i, "train_score" : accuracy_score(y_train, y_train_predict), "test_score" : accuracy_score(y_test, y_test_predict)}
        results.append(result)
        models.append(model)
    return sorted(results, key = lambda x : x["train_score"], reverse=True), models

results, models = Training(traindfs, testdfs)

x_train_sel_without_outliers_standarscale

#Creamos el diccionario de hiperparametros
hyperparameters = { "C" : np.linspace(0.0, 5.0, num=10), "penalty" : ["l1", "l2", "elasticnet", "None"], "solver" : ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]}

hyperparameters

#Pasamos el modelo preentrenado con los hiperparametros (En este caso paso un grid pero podría pasar un RandomSearchCV)
grid = GridSearchCV(models[1], hyperparameters, scoring="accuracy")

grid.fit(x_train_sel_without_outliers_standarscale, y_train)

#Guardamos los mejores parametros

grid.best_params_

#Guardamos el modelo entrenado

best_model = grid.best_estimator_

y_test_predict = best_model.predict(x_test_sel_without_outliers_standarscale)

accuracy_score(y_test, y_test_predict)

#Creamos la matriz de confusion para ver donde estaban los errores

confmatrx = pd.DataFrame(confusion_matrix(y_test, y_test_predict))

plt.figure(figsize=(5,5))

sns.heatmap(data = confmatrx, annot = True, fmt="d", cbar = False)

plt.show()

#Creamos la matriz de confusion para ver donde estaban los errores

confmatrx = pd.DataFrame(confusion_matrix(y_test, y_test_predict))

plt.figure(figsize=(5,5))

sns.heatmap(data = confmatrx, annot = True, fmt="d", cbar = False)

plt.show()
