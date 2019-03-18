# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:09:11 2019

@author: dreth
"""

from alpha_vantage.timeseries import TimeSeries as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
from scipy.optimize import minimize
import random as rd

# Arreglando el layout de los gráficos para que los labels del eje x (que tienen gran longitud)
# queden propiamente puestos en la ventana de figura.
rcParams.update({'figure.autolayout': True})

# API key para utilizar alpha_vantage
ts = ts(key='ZN5W0AUQ2LWX8JTD', output_format='pandas'); data = []

# tasa libre de riesgo
TSL = 0.04

# simbolos a importar
syms = ['DJI','ABBV','SLG','PNW','NEM','DIS','NVDA','RI.PA','DUK','LUV','PLD']

# iteraciones
iteraciones = 25

# funcion para obtener la tabla de datos historicos diarios de los simbolos seleccionados
# symbol es una lista de datos con los simbolos de las compañias en string y siendo el
# primer elemento un indice bursatil seleccionado para luego calcular beta y alfa
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
                
                # en el 100% de las veces que se corre alpha_vantage para muchos simbolos, se genera
                # un error por enviar muchos requests al servidor, es normal
                # con esto agarramos el error y podemos reintentar la ejecucion
                except KeyError:
                    print('Error de alpha_vantage, reintentando')
                    continue
                break
        # Convirtiendo el dict a un DataFrame
        symdf = pd.DataFrame(symbol_dict)
        
        # Limpiando la tabla de datos
        symdf = symdf.dropna()
        
    # Agarre de errores con mensaje de error para símbolos incorrectos
    except (OverflowError):
        print('ℹ️ Símbolo introducido incorrectamente, error.')
        return
    
    # convirtiendo los indices de la tabla de datos en fechas
    symdf.index = pd.to_datetime(symdf.index)
    return symdf

# obteniendo la informacion historica de los simbolos definidos arriba
Data = get_symdf(syms)

# el primer elemento en el dataframe de los stocks debe ser el indice bursatil para calcular beta etc
# recibe como parametro df que es la tabla de datos con la informacion diaria
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
  
# Generando las tablas con los ratios      
Ratios = ratios(Data)

# generando la matriz de correlaciones
correl = Data[Data.columns[1:]].corr()

# creando una funcion para calcular la matriz de covarianzas
def matriz_cov(corr,ratiosdf):
    
    # creando diccionario de covarianzas para calcular
    cov = {}

    for sym in corr.columns:
        for std in ratiosdf.iloc[1:]['std']:
            iterlst = []
            
            # productos de las desviaciones estándar
            stdprod = [x*std for x in ratiosdf.iloc[1:]['std']]
            for x, y in zip(corr[sym],stdprod):
                iterlst.append(x*y)
        cov[sym] = iterlst
    return cov

# generando la matriz de covarianzas
cov = matriz_cov(correl,Ratios[1])

# creando el dataframe con las covarianzas para poder obtener la matriz de covarianzas
cov = pd.DataFrame(cov)
cov.index = cov.columns

# creando una lista con los rendimientos para utilizar en el solver
lst = [x for x in Ratios[1].iloc[1:].rendimiento]

# funcion del optimizador, siendo it = cantidad de iteraciones
def solver(it):
    
    # creando la lista de datos igualmente espaciados de los rendimientos
    linspace = np.linspace(TSL, max(lst), num = it-1)
    
    # creando una tupla con todos los limites superiores e inferiores para cada variable 
    b = tuple([(0,1) for x in range(len(lst))])
    
    # defininiendo la lista para generar las soluciones
    solutions = []
    
    for i in range(0,it-1):
        
        # defininiendo la funcion objetivo (todas las variables a utilizar)
        # cada x con un numero es un porcentaje de la cartera a asignar a cada simbolo
        def objective_function(x):
            x1 = x[0]; x2 = x[1]; x3 = x[2]
            x4 = x[3]; x5 = x[4]; x6 = x[5]
            x7 = x[6]; x8 = x[7]; x9 = x[8]
            x10 = x[9]
            return lst[0]*x1 + lst[1]*x2 + lst[2]*x3 + lst[3]*x4 + lst[4]*x5 + lst[5]*x6 + lst[6]*x7 + lst[7]*x8 + lst[8]*x9 + lst[9]*x10

        # defininiendo los constraints del problema
        def constraint1(x):
            return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] - 1
        
        # definiendo el constraint del problema que determina los valores de las variables para cada rendimiento asignado
        def constraint2(x):
            return lst[0]*x[0] + lst[1]*x[1] + lst[2]*x[2] + lst[3]*x[3] + lst[4]*x[4] + lst[5]*x[5] + lst[6]*x[6] + lst[7]*x[7] + lst[8]*x[8] + lst[9]*x[9] - linspace[i]
        
        # indicando los parámetros del problema para el optimizador
        con1 = {'type': 'eq', 'fun': constraint1}
        con2 = {'type': 'eq', 'fun': constraint2}
        
        # definiendo valores iniciales, que no importan mucho realmente mientras se encuentren en el intervalo [0,1]
        x0 = [rd.random() for x in range(len(Ratios[1])-1)]
        
        # utilizando el solver
        sol = minimize(objective_function,x0,method='SLSQP',bounds=b,constraints=[con1,con2])
        
        # agregando cada una de las solciones a una lista para imprimir como solución
        # el primer elemento de cada lista es el rendimiento a esperarse de cada
        # conjunto de soluciones
        solutions.append([linspace[i],sol.x])
        
    # retornando las soluciones
    return solutions

# Utilizando el solver con la cantidad de simbolos que tenemos sin contar el indice bursatil
solver_solution = solver(iteraciones)
 
# creando una función para la varianza del E(RP)
# siendo dfcov = matriz de covarianzas
# siendo sol = solucion del solver
def varianzaERP(dfcov, sol):
    
    # inicializando la varianza en 0 para cada ciclo
    var = 0
    
    # seleccionando los datos de la solucion del solver eliminando la rentabilidad de la lista
    # ya que este elemento no se utilizará para los cálculos
    sol = [(x[1]**2).tolist() for x in sol]
    
    # creando una lista para todas las varianzas correspondientes
    lista_var = []
    
    # bucle para calcular la varianza que cicla primero entre cada columna del df, luego entre
    # cada elemento de cada columna y luego en un list comprehension para calcular el producto
    # del valor de cada columna con el valor de cada uno de los porcentajes asignados a cada 
    # simbolo y luego insertandolo en una lista 
    for k in sol:
        for col in dfcov:
            for d, elem in zip(k,dfcov[col]):
                val = d*elem
                var = val + var
        lista_var.append(var)
    return lista_var

# definiendo la varianza utilizando la función de varianzaE_RP
varERP = varianzaERP(cov, solver_solution)

# agregando la varianza a cada conjunto solucion de datos del solver con su rendimiento correspondiente
varlist = varERP

# generando listas con los demas elementos para crear el df con los E(RP) de cada rentabilidad
solutionlist = [x[0].tolist() for x in solver_solution]

# creando la lista con las desviaciones de cada una de las iteraciones
stdlist = [np.sqrt(k) for k in varlist]

# creando la lista con los betas de cada una de las iteraciones
beta_to_add = []; betalist = []
for solution in [x[1].tolist() for x in solver_solution]:
    beta_per_iter = []
    for point, b in zip(solution, Ratios[1].iloc[1:].beta):
        beta_per_iter.append(point*b)
    betalist.append(sum(beta_per_iter))

# eliminando la lista utilizada para no ocupar memoria en variables innecesarias
del beta_to_add

# creando la lista con los sharpe de cada iteracion
sharpelist = [(x-TSL)/y for x, y in zip([x[0] for x in solver_solution],stdlist)]

# adjuntando todas las variables de cada columna a una sola lista
columns_portafolio = [solutionlist, stdlist, sharpelist, betalist]

# creando el diccionario para generar el df y adjuntando cada elemento
# de cada una de las variables a las columnas de las iteraciones en el df
portafolio_dict = {}; columns_df = ['rentabilidad','std','sharpe','beta']
for x, y in zip(columns_df,columns_portafolio):
    portafolio_dict[x] = y 

# creando un df para la lista de iteraciones
solver_iter = [x[1].tolist() for x in solver_solution]; solver_iter_dict = {}
for i in range(len(solver_iter)):
    solver_iter_dict[i] = solver_iter[i]
solver_iter = pd.DataFrame(solver_iter_dict)
solver_iter.index = syms[1:]

# generando el df del portafolio a partir del diccionario creado
portafolio = pd.DataFrame(portafolio_dict)

# generando el grafico del portafolio
portafolio[['std','rentabilidad']].plot(kind='line',x='std',y='rentabilidad')
portafolio[['std','rentabilidad']].plot(kind='scatter',x='std',y='rentabilidad')
plt.show()



