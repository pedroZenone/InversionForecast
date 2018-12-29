# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:19:30 2018

@author: pedzenon
"""

import pandas as pd
import os
from os.path import isfile, join,dirname
import statsmodels.api as sm

fileDir = os.path.dirname(os.path.realpath('__file__'))
fileIn= os.path.join(fileDir, 'data')


import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sn
import numpy as np
import dateutil.relativedelta
from scipy.spatial.distance import mahalanobis
import scipy as sp
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# Devuelve la operacion binaria entre x.marca == colss_eval & categoria == x.categoria
# Como se admiten muchas marcas (colss_eval), se evalua cada x.marca & categoria 
def binary_gen(x,colss_eval,categoria):
    
    cols_eval = colss_eval.copy()
    binary = (x.MARCA == cols_eval[0]) & (x.CATEGORIA == categoria)
    # Si solo entro un dato, retorno el resultado incial
    if(len(cols_eval) == 1):
        return binary
    
    cols_eval.pop(0)
    
    for col in cols_eval:
        binary = binary | ((x.MARCA == col) & (x.CATEGORIA == categoria))
    return binary

# Funcion base de mahalanobis, para sacar outliers. Ojo: X no debe tener dummies ni nans!
def mahalanobisR(X,meanCol,IC):
    print(X.shape)
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.ix[i,:],meanCol,IC) ** 2)
    return(m)

# @Funcion: Mahalanobis. 
# @param cols. son las columnas de data que te va a levantar.
# @param Debug. es un boolean: Si es True sólo te muestra el histograma de distancias para que
#   puedas setear el tresh indicado. (tenes que elegir un valor donde en el histograma tienda a 0, el resto seran outliers)
# @param tresh: es el treshold de corte
def MahalaRecorte(data,cols,debug,tresh):    
    analisis = data[cols].reset_index(drop = True)
    Sx = analisis.cov().values
    Sx = sp.linalg.inv(Sx)
    
    mean = analisis.mean().values
    
    mR = mahalanobisR(analisis,mean,Sx)
    if(debug == True):
        plt.hist(mR)   # aca defino el treshold de outliers!
        return data
    else:
        data["mR"] = mR
        data = data.loc[data.mR < tresh]
        return data.drop(["mR"],axis = 1)

# Esta funcion te permite shiftear la serie de tiempo
# @param inversion: dataset
# @para cols: son las columnas que queres agregar. ojo con le orden ya que si agregas
#        3 columnas, significa que queres 3 años previos
# @param terget: nombre de la columna que queres shiftear en tiempo        
def add_previous(inversion,cols,target):
    
    inversion = inversion.copy()
    
    for col in cols:
        inversion[col] = np.nan
 
    for ind,line in inversion.iterrows():
        for i,col in enumerate(cols):
            date = line.ANO - (i+1)          
    
            aux = inversion.loc[(inversion.ANO == date)]
            if (aux.shape[0]):
                inversion.loc[ind,col] = aux[target].iloc[0]    
    
    return inversion           
        

def month_sub(row,sub_month ):
#    sub_month = 1
    result_month = 0
    result_year = 0
    year = row["ANO"] ; month = row["MES"]
    if month > (sub_month % 12):
        result_month = month - (sub_month % 12)
        result_year = year - (sub_month / 12)
    else:
        result_month = 12 - (sub_month % 12) + month
        result_year = year - (sub_month / 12 + 1)
    row["ANO"] = int(result_year) + 1
    row["MES"] = result_month
    return row

# Imputa el campo value con la media agrupada por group
def my_imputer(df,value,group):
    df[value] = df[value].replace(np.inf, np.nan)
    df[value] = df[value].replace(NaN, np.nan)
    df[value] = df[value].fillna(df.loc[df.ANO < 2014][value].mean())
    return df

# Funcion que uso para cambiar el nombre de una lista en un vector. Util para cmabiar el nombre de las columnas de un dataframe
def my_columnReplace(cols,which,replace):
    resu = []
    for i,col in enumerate(cols):
        if(col == which):
            resu.append(replace)
        else:
            resu.append(col)
    return resu

# Genero la sumarizacion del Q1
def first_resume(x):
    return x.loc[x.MES < 4]["INVERSION"].sum()

# Genero la sumarizacion del Q2
def middle_resume(x):
    return x.loc[(x.MES >= 4) & (x.MES <= 9)]["INVERSION"].sum()

# Genero la sumarizacion del Q3
def last_resume(x):
    return x.loc[x.MES >= 10]["INVERSION"].sum()

########################################################################################
#####################       Main:   ##################################################
    
# IPC_byAno: me genero un dataframe de inflacion sumarizada por año y con una columna que representa la multiplicatoria
# de la inflacion acumulada
IPC = pd.read_excel(fileIn + '\\'+ "IPCcongreso.xlsx")
IPC["ANO"] = IPC.Mes.apply(lambda x: x.year)
IPC["MES"] = IPC.Mes.apply(lambda x: x.month)
IPC_byAno = IPC.groupby(['ANO'])[['IPCcongreso']].sum()
IPC_byAno.loc[IPC_byAno.index == 2018,"IPCcongreso"] = 29  # estimo una inflacion de 29%
IPC_byAno["acum"] = 0.0

for ano in range(2009,2019):
    inf = IPC_byAno.loc[IPC_byAno.index <= ano]["IPCcongreso"].tolist()
    aux = 1
    for i in inf:
        aux = aux*(1+i/100)
    IPC_byAno.loc[IPC_byAno.index == ano,"acum"] = aux
    
IPC_byAno["ANO"] = IPC_byAno.index


# Inversion. Empiezo a masajear la data
inversion = pd.read_csv(fileIn + '\\'+ 'IBOPEinversion_depurado3.csv',sep = ';',encoding = 'ANSI')

# desafecto por la inflacion. Divido la inversion por la inflacion acumulada
def IPC_off(x):
    return x["INVERSION"]/x["acum"]

inversion = pd.merge(inversion,IPC_byAno,on=["ANO"])
inversion["INVERSIONdesafectada"] = inversion.apply(IPC_off,axis = 1)
inversion["INVERSION"] = inversion["INVERSIONdesafectada"]

# desagrego el campo MEDIO ya que el analisis no tiene en cuenta el medio!
inversion = inversion.groupby(['DATE','ANO','MES','CATEGORIA','MARCA','ANUNCIANTE','SUBCATEGORIA','PRIORIDAD'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
inversion = inversion.reset_index()

# saco el outlier
inversion = inversion.loc[~((inversion.MARCA == "OMMITED_BRAND") & (inversion.ANO == 2011))]

# @brief: competencia_generator te genera un dataframe con la competencia y features agregados.
# @param marcas_competencia: lista de marcas de la competencia.
# @param categoria: categoria a la que pertenecen esa lista de marcas de competencia
# @param output: una vez que se selecciona el dataframe de competencias se agregan features como: inversion de 1 y 2 años pasados, la inversion en los Q1,Q2,Q3 del año acutal y los 2 añlos pasados. Tambien se generan transformaciones (log,sqrt,cuadrado) de 2 años anteriores, GPRS, Segmento promedio, CPR y variables macro 
def competencia_generator(marcas_competencia,categoria,IPC):  
    competencia = inversion.loc[binary_gen(inversion,marcas_competencia,categoria)]
    
    # Primero filtro por competencia y sumo todas las marcas mensualmente
    competencia = competencia.groupby(['ANO','MES'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
    competencia = competencia.reset_index()
    # me creo las metricas descriptivas del año
    aux_first = pd.DataFrame(competencia.groupby(['ANO'],axis = 0).apply(first_resume),columns = ["INVERSION_FIRST"])
    aux_first = aux_first.reset_index()
    aux_middle = pd.DataFrame(competencia.groupby(['ANO'],axis = 0).apply(middle_resume),columns = ["INVERSION_MIDDLE"])
    aux_middle = aux_middle.reset_index()
    aux_last = pd.DataFrame(competencia.groupby(['ANO'],axis = 0).apply(last_resume),columns = ["INVERSION_LAST"])
    aux_last = aux_last.reset_index()
    # Agrupo todo los años
    competencia = competencia.groupby(['ANO'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
    competencia = competencia.reset_index()
    
    # le agrego las metricas agregadas
    competencia = pd.merge(competencia,aux_first, how = "left", on=['ANO'])
    competencia = pd.merge(competencia,aux_middle, how = "left", on=['ANO'])
    competencia = pd.merge(competencia,aux_last, how = "left", on=['ANO'])
    
    # Agrego distintas medidas agregadas
    competencia["SegProm"] = competencia["VOL_FISICO"]/competencia["CANT_AVISOS"]
    competencia.columns = my_columnReplace(competencia.columns,"INVERSION","target")
    competencia = add_previous(competencia,["INVERSION1","INVERSION2"],"target")
    competencia = add_previous(competencia,["CANT_AVISOS1","CANT_AVISOS2"],"CANT_AVISOS").drop(["CANT_AVISOS"],axis = 1)
    competencia = add_previous(competencia,["VOL_FISICO1","VOL_FISICO2"],"VOL_FISICO").drop(["VOL_FISICO"],axis = 1)
    competencia = add_previous(competencia,["SegProm1","SegProm2"],"SegProm").drop(["SegProm"],axis = 1)
    competencia["mundial"] = competencia.ANO.apply(lambda x: 1 if ((x % 2)== 0) else 0) # los mundiales y juegos olimpicos se dan en los años pares
    competencia["logInv1"] = np.log(competencia["INVERSION1"])
    competencia["Inv2_1"] = competencia["INVERSION1"]*competencia["INVERSION1"]
    competencia["sqrtInv1"] = np.sqrt(competencia["INVERSION1"])
    competencia["logInv1ANDVOL"] = competencia["logInv1"]/competencia["VOL_FISICO1"]
    competencia = add_previous(competencia,["INVERSION_FIRST1","INVERSION_FIRST2"],"INVERSION_FIRST")
    competencia = add_previous(competencia,["INVERSION_MIDDLE1","INVERSION_MIDDLE2"],"INVERSION_MIDDLE").drop(["INVERSION_MIDDLE"],axis = 1)
    competencia = add_previous(competencia,["INVERSION_LAST1","INVERSION_LAST2"],"INVERSION_LAST").drop(["INVERSION_LAST"],axis = 1)
    
    # le agrego GRPS. Primero lo corro 1 mes para que pueda aprender del pasado
    GRPS = pd.read_csv(fileIn + '\\'+"GRPS_2009a2018.csv",sep = ";",encoding = "ANSI")
    GRPS = GRPS.loc[binary_gen(GRPS,marcas_competencia,categoria)]
    GRPS = GRPS.groupby(['ANO'])[['UNIVERSE_TVA','UNIVERSE_TVC']].sum()
    GRPS = GRPS.reset_index()
    GRPS["UNIVERSE"] = GRPS["UNIVERSE_TVA"] + GRPS["UNIVERSE_TVC"]
    GRPS = add_previous(GRPS,["UNIVERSE1","UNIVERSE2"],"UNIVERSE").drop(["UNIVERSE"],axis = 1)
    competencia = pd.merge(competencia,GRPS[['ANO',"UNIVERSE1","UNIVERSE2"]],on = ['ANO'], how = "left")
    competencia["CPR1"] = competencia["INVERSION1"]/competencia["UNIVERSE1"]
    competencia["CPR2"] = competencia["INVERSION2"]/competencia["UNIVERSE2"]
    competencia["Inv_avisos1"] = competencia["INVERSION1"]/competencia["CANT_AVISOS1"]
    competencia["Inv_avisos2"] = competencia["INVERSION2"]/competencia["CANT_AVISOS2"]
    
    # agrego variables macro
    preci = pd.read_csv(fileIn + '\\'+"precipitaciones_historico.csv",sep = ";")
    preci = preci.groupby(["ANO"])["Precipitaciones"].sum().reset_index()
    temp = pd.read_csv(fileIn + '\\'+"temperatura_historico.csv",sep = ";")
    temp = temp.groupby(["ANO"])[['tempMax', 'tempMin', 'tempAvg']].mean().reset_index()
    PBI = pd.read_csv(fileIn + '\\'+"PBI_historico.csv",sep = ";")
    PBI = PBI.groupby(["ANO"])["PBI"].sum().reset_index()
    IPC = IPC.groupby(["ANO"])["IPCcongreso"].sum().reset_index()
    poblacion = pd.read_csv(fileIn + '\\'+"poblacion_historico.csv",sep = ";")
    inflacion_medios = pd.read_csv(fileIn + '\\'+"inflacion_medios_historico.csv",sep = ";")
    competencia = pd.merge(competencia,preci,on=["ANO"])
    competencia = pd.merge(competencia,temp,on = ['ANO'])
    competencia = pd.merge(competencia,IPC[["IPCcongreso","ANO"]],on=["ANO"])
    competencia = pd.merge(competencia,PBI,on = ['ANO'])
    competencia = pd.merge(competencia,poblacion,on = ['ANO'])
    competencia = pd.merge(competencia,inflacion_medios,on = ['ANO'])
    
    return competencia

# comienzo a generarme el dataframe apendeado con todas las competencias
competencia1 = competencia_generator(['OMMITED_BRAND'],'SSDs - COLAS',IPC.copy())
competencia1 = competencia1.append(competencia_generator(['OMMITED_BRAND'],'SSDs - FLAVORS',IPC.copy()),ignore_index = True)
competencia1 = competencia1.append(competencia_generator(['OMMITED_BRAND'],'PLAIN WATER',IPC.copy()),ignore_index = True)
competencia1 = competencia1.append(competencia_generator(['OMMITED_BRAND'],'JUGOS',IPC.copy()),ignore_index = True)
competencia1 = competencia1.append(competencia_generator(['OMMITED_BRAND'],'ISOTONICS',IPC.copy()),ignore_index = True)
competencia1 = competencia1.append(competencia_generator(['OMMITED_BRAND'],'FLAVOURED WATER',IPC.copy()),ignore_index = True)

# Le saco 2018 y 2009 ya que no tienen data potable par ala prediccion (muchos nan)
competencia = competencia1.loc[competencia1.ANO < 2018]
competencia = competencia.loc[competencia.ANO > 2009]  
competencia = competencia.loc[competencia.VOL_FISICO2 != OMMITED_NUMBER] # Saco outlier
 
cor_matrix = competencia.corr()  # evaluo correlaciones

# hago un estudio de mahalanobis para detectar outliers

# Estas son todas las columnas potables que se evaluan en el modelo
cols = ['target','INVERSION_FIRST','INVERSION_FIRST2','INVERSION_FIRST1','INVERSION_LAST2','INVERSION_LAST1','INVERSION_MIDDLE1','INVERSION_MIDDLE2','logInv1','INVERSION1','INVERSION2','sqrtInv1','VOL_FISICO2','VOL_FISICO1']

# imputo por la media agrupada por año y columna    
for col in cols:
    competencia = my_imputer(competencia,col,'ANO')
    
competencia = competencia[cols].dropna()
cols_mahala = ['VOL_FISICO2', 'INVERSION_MIDDLE2', 'INVERSION_LAST2','INVERSION2', 'INVERSION1']
competencia = MahalaRecorte(competencia,cols_mahala,False,15) # aca descarto aoutliers

cols =  ['VOL_FISICO2', 'INVERSION_MIDDLE2', 'INVERSION_LAST2','INVERSION2', 'INVERSION1']
# Le agrego las dumies
#dummies = ['finMes','mundialEspecifico']
##cols = cols + dummies
#competencia = competencia[cols]
#competencia["INVERSION1_2"] = competencia["INVERSION1"]*competencia["INVERSION1"]
#competencia["INVERSION1sqrt"] = competencia["INVERSION1"].apply(lambda x: np.sqrt(x))

# Me armo un array de todas las combinaciones posibles de features
import itertools
combinations = []
cols_iter = cols[1:]
for L in range(0, len(cols)+1):
    for subset in itertools.combinations(cols, L):
        aux = list(subset)
        aux.append('target')
        combinations.append(aux)

combinations.pop(0);combinations.pop(0)
resu = []

# evaluo los r2 de todas las combinaciones
for i,comb in enumerate(combinations):

    X = competencia[comb].drop(["target"],axis = 1)
    #X = cctm.drop(['INVERSION_y'],axis = 1)
    y = competencia["target"]
    model = LinearRegression().fit(X,y)
    aux = {}
    aux["score"] = r2_score(y,model.predict(X))
    aux["param"] = comb
    resu.append(aux)

resu = pd.DataFrame(resu)   


X = StandardScaler().fit_transform(X)

# Ok. Ya elegi la combinacion que mas me gusto. Ahora veo relevancia estadistica de cada variable


cols_import = ['VOL_FISICO2', 'INVERSION_MIDDLE2', 'INVERSION2', 'INVERSION1']

X = competencia[cols_import]
X = StandardScaler().fit_transform(X)
X = PCA(n_components = 2).fit_transform(X)
X = sm.add_constant(X)
y = competencia["target"]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print (model.summary())

# Ahora me armo el modelo
cols_import = ['VOL_FISICO2', 'INVERSION_MIDDLE2', 'INVERSION2', 'INVERSION1']

X = competencia[cols_import]
pipe = Pipeline(steps=[('Scale', StandardScaler()), ('PCA', PCA(n_components = 2)), ('LR', LinearRegression())])
y = competencia["target"]
modelo = pipe.fit(X,y)
#from sklearn.externals import joblib
#joblib.dump(modelo, 'modelo_Anual.pkl') 

# comparo las predicciones con los SOI de los años anteriores
marcas_competencia = OMMITED_BRAND
categorias = ['SSDs - COLAS','SSDs - FLAVORS','PLAIN WATER','JUGOS','FLAVOURED WATER','ISOTONICS']
marcas_KO = OMMITED_BRAND

for i in range(len(marcas_competencia)): 
    competenciaFinal = competencia_generator(marcas_competencia[i],categorias[i],IPC.copy())
    competenciaFinal = competenciaFinal.loc[competenciaFinal.ANO == 2018]
    anterior = inversion.loc[binary_gen(inversion,marcas_competencia[i],categorias[i])]
    anterior = anterior.loc[anterior.ANO == 2017]["INVERSION"].sum()
    predict_2018 = modelo.predict(competenciaFinal[cols_import])[0]
    KO = inversion.loc[binary_gen(inversion,marcas_KO[i],categorias[i])]
    KO = KO.loc[KO.ANO == 2017]["INVERSION"].sum()
    print(marcas_KO[i],'. SOI: ',KO/(KO+predict_2018), 'predije: ',predict_2018)
    print("anterior 2017: ",KO/(KO+anterior))
    print("KO: ",KO)
    anterior = inversion.loc[binary_gen(inversion,marcas_competencia[i],categorias[i])]
    anterior = anterior.loc[anterior.ANO == 2016]["INVERSION"].sum()
    KO = inversion.loc[binary_gen(inversion,marcas_KO[i],categorias[i])]
    KO = KO.loc[KO.ANO == 2016]["INVERSION"].sum()
    print("Anterior 2016: ",KO/(KO+anterior))
    anterior = inversion.loc[binary_gen(inversion,marcas_competencia[i],categorias[i])]
    anterior = anterior.loc[anterior.ANO == 2015]["INVERSION"].sum()
    KO = inversion.loc[binary_gen(inversion,marcas_KO[i],categorias[i])]
    KO = KO.loc[KO.ANO == 2015]["INVERSION"].sum()
    print("Anterior 2015: ",KO/(KO+anterior))
    anterior = inversion.loc[binary_gen(inversion,marcas_competencia[i],categorias[i])]
    anterior = anterior.loc[anterior.ANO == 2014]["INVERSION"].sum()
    KO = inversion.loc[binary_gen(inversion,marcas_KO[i],categorias[i])]
    KO = KO.loc[KO.ANO == 2014]["INVERSION"].sum()
    print("Anterior 2014: ",KO/(KO+anterior))
    
######################################################################################################

#Para 2019:

def armar_data(x,year,inv1):

    aux_middle = pd.DataFrame(x.groupby(['ANO'],axis = 0).apply(first_resume),columns = ["INVERSION_MIDDLE"])
    aux_middle = aux_middle.reset_index()

    
    x = x.groupby(['ANO'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
    x = x.reset_index()
    
    x = pd.merge(x,aux_middle, how = "left", on=['ANO'])
    
    if(inv1 == 0):
        inv1 =x.loc[x.ANO == (year-1)]["INVERSION"].iloc[0]
    else:
        inv1 = inv1

    inv_middle = x.loc[x.ANO == (year-2)]["INVERSION_MIDDLE"].iloc[0]
    inv2 =x.loc[x.ANO == (year-2)]["INVERSION"].iloc[0]
    vol_f = x.loc[x.ANO == (year-2)]["VOL_FISICO"].iloc[0]
   
    return pd.DataFrame([{'VOL_FISICO2':vol_f,'INVERSION_MIDDLE2':inv_middle,'INVERSION1':inv1,'INVERSION2':inv2}])[['VOL_FISICO2', 'INVERSION_MIDDLE2', 'INVERSION2', 'INVERSION1']]

marcas_competencia = OMMITED_BRAND
categorias = ['SSDs - COLAS','SSDs - FLAVORS','PLAIN WATER','JUGOS','FLAVOURED WATER','ISOTONICS']
marcas_KO = OMMITED_BRAND

marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'SSDs - COLAS')]
competenciaFinal = armar_data(competenciaFinal,2019,OMMITED_NUMBER)
print(marca, ": ",modelo.predict(competenciaFinal)[0])

marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'SSDs - FLAVORS')]
competenciaFinal = armar_data(competenciaFinal,2019,OMMITED_NUMBER)
print(marca, ": ",modelo.predict(competenciaFinal)[0])

marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'PLAIN WATER')]
competenciaFinal = armar_data(competenciaFinal,2019,OMMITED_NUMBER)
print(marca, ": ",modelo.predict(competenciaFinal)[0])

marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'JUGOS')]
competenciaFinal = armar_data(competenciaFinal,2019,OMMITED_NUMBER)
print(marca, ": ",modelo.predict(competenciaFinal)[0])

marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'ISOTONICS')]
competenciaFinal = armar_data(competenciaFinal,2018,0)
print(marca, ": ",modelo.predict(competenciaFinal)[0])
marca = OMMITED_BRAND
competenciaFinal = inversion.loc[binary_gen(inversion,marca,'ISOTONICS')]
competenciaFinal = armar_data(competenciaFinal,2019,OMMITED_NUMBER)
print(marca, ": ",modelo.predict(competenciaFinal)[0])

competenciaFinal = pd.read_csv(fileIn + '\\'+ "data_flavoured.csv",sep = ";")
competenciaFinal = competenciaFinal.groupby(['ANO','MES'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
competenciaFinal = competenciaFinal.reset_index()
print('FLAVOURED: ',modelo.predict(armar_data(competenciaFinal,2018,0) ))
print('FLAVOURED: ',modelo.predict(armar_data(competenciaFinal,2019,OMMITED_NUMBER) ))
competenciaFinal = competenciaFinal.groupby(['ANO'])[['VOL_FISICO','CANT_AVISOS','INVERSION']].sum()
competenciaFinal = competenciaFinal.reset_index()
