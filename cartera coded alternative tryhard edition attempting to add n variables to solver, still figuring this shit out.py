# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:09:11 2019

@author: dreth
"""

from alpha_vantage.timeseries import TimeSeries as ts
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from time import sleep
from scipy.stats import linregress
from scipy.optimize import minimize

# Arreglando el layout de los gráficos para que los labels del eje x (que tienen gran longitud)
# queden propiamente puestos en la ventana de figura.
rcParams.update({'figure.autolayout': True})

# API key para utilizar alpha_vantage
ts = ts(key='ZN5W0AUQ2LWX8JTD', output_format='pandas'); data = []

TSL = 0.04

def get_symdf(symbol):
    
    # creando el diccionario de data para el df
    symbol_dict = {}
    count = 0
    
    # Chequeo para determinar si el usuario introdujo correctamente el símbolo de la compañía
    try: 
        for sym in symbol:
            count += 1
            while True:
                # Probando convertir el símbolo a string
                (str(sym))
                
                try:
                    # Adquisición de datos
                    data, meta_data = ts.get_daily(symbol = sym, outputsize = 'full')
                    symbol_dict[sym] = data['4. close']

                except KeyError:
                    print('Error de alpha_vantage, reintentando')
                    continue
                break
        # Convirtiendo el dict a un DataFrame
        symdf = pd.DataFrame(symbol_dict)
        
        # Limpiando el df
        symdf = symdf.dropna()
        
    # Agarre de errores con mensaje de error para símbolos incorrectos
    except (OverflowError):
        print('ℹ️ Símbolo introducido incorrectamente, error.')
        return
    
    symdf.index = pd.to_datetime(symdf.index)
    return symdf

# el primer elemento en el dataframe de los stocks debe ser el indice bursatil para calcular beta etc
def ratios(df):
    
    #generando un diccionario para convetir a df
    ratios_dict = {}
    valores = ['rendimiento', 'var', 'std', 'cv', 'beta', 'alfa', 'sharpe', 'treynor', 'max', 'min']
    valores_relevantes = {}
    for val in valores:
        valores_relevantes[val] = []
    
    #loop para crear los ratios en escala logaritmica
    for sym in df:
        ratios = []
        for i in range(len(df[sym])-1):
            ratios.append(np.log(df[sym][i+1]/df[sym][i]))
        ratios_dict[sym] = ratios
    
    #creando el df con los ratios
    ratios_df = pd.DataFrame(ratios_dict)
    
    #calculando valores relevantes (rend, var, etc)
    for sym in ratios_df:
        
        # Rendimiento
        rendimiento = np.average(ratios_df[sym])*252
        valores_relevantes[valores[0]].append(rendimiento)
        
        # Var
        var = np.var(ratios_df[sym])
        valores_relevantes[valores[1]].append(var)
        
        # Std
        std = np.std(ratios_df[sym])
        valores_relevantes[valores[2]].append(std)
        
        # Beta y alfa
        beta_alfa = list(linregress(ratios_df[ratios_df.columns[0]],ratios_df[sym]))
        valores_relevantes[valores[4]].append(beta_alfa[0])
        valores_relevantes[valores[5]].append(beta_alfa[1])
        
        # Max y Min
        max_min = [np.max(ratios_df[sym]), np.min(ratios_df[sym])]
        valores_relevantes[valores[8]].append(max_min[0])
        valores_relevantes[valores[9]].append(max_min[1])
      
    # CV y Sharpe
    for x, y in zip(valores_relevantes['std'], valores_relevantes['rendimiento']):
        
        # CV
        cv = x/y
        valores_relevantes[valores[3]].append(cv)
        
        # Sharpe
        sharpe = (y - TSL) / x
        valores_relevantes[valores[6]].append(sharpe)

    # Treynor
    for x, y in zip(valores_relevantes['rendimiento'], valores_relevantes['beta']):
        treynor = (x - TSL) / y
        valores_relevantes[valores[7]].append(treynor)
    
    valores_relevantes = pd.DataFrame(valores_relevantes)
    valores_relevantes.index = ratios_df.columns
    
    return (ratios_df,valores_relevantes)
  
# Todas las tablas a utilizar definidas         
Data = get_symdf(['DJI','ABBV','BP','PFE','CF','DIS','NVDA','RI.PA','DUK','LUV','PLD'])
Ratios = ratios(Data)
correl = Data[Data.columns[1:]].corr()

# creando diccionario de covarianzas para calcular
cov = {}

# Calculando Matriz de covarianzas
for sym in correl.columns:
    for std in Ratios[1].iloc[1:]['std']:
        iterlst = []
        # productos de las desviaciones estándar
        stdprod = [x*std for x in Ratios[1]['std']]
        for x, y in zip(correl[sym],stdprod):
            iterlst.append(x*y)
            print(iterlst)
    cov[sym] = iterlst

# creando el dataframe con las covarianzas para poder obtener la matriz de covarianzas
cov = pd.DataFrame(cov)
cov.index = cov.columns

# creando una lista con los rendimientos para utilizar en el solver
lst = [x for x in Ratios[1].iloc[1:].rendimiento]

def solver(varnum, it):
    xvar = {}
    linspace = np.linspace(TSL, max(Ratios[1].iloc[1:].rendimiento), num =it)
    b = tuple([(0,1) for x in range(varnum)])
    solutions = []
    for i in range(0,it-1):
        
        def objective_function(x):
            for k in range(0,varnum-1):
                xvar['x{0}'.format(k)] = x[k]
            print(xvar)
            zipobj = zip([key for key, value in xvar],[value for key, value in xvar])
            result = [x*y for x, y in zipobj]
            
            return  sum(result)
        
        def constraint1(x):
            result = [value for key, value in xvar]
            return sum(result)-1
        
        def constraint2(x):
            for s in range(0,varnum-1):
                zipobj = zip([value for key, value in xvar],[value for key, value in xvar])
            result = [x*y for x,y in zipobj]
            
            return sum(result) - linspace[i]
        
        con1 = {'type': 'eq', 'fun': constraint1}
        con2 = {'type': 'eq', 'fun': constraint2}
        
        x0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        sol = minimize(objective_function,x0,method='SLSQP',bounds=b,constraints=[con1,con2])
        solutions.append(sol.x)
        
        
        
            
            