# -*- coding: utf-8 -*-
"""
@author: Antonio Javier
"""

import numpy as np
import pandas as pd
import time
import itertools
import argparse

from scipy.sparse import csr_matrix as csr
#from scipy.sparse import random

# Dependiendo de como se configure Spark las siguientes lineas pueden ser necesarias.
#import findspark
#findspark.init('/home/antoniojavier/spark-2.4.4-bin-hadoop2.7')
import pyspark as spark
#from pyspark import *
from pyspark.sql import *

""" Funciones necesarias para el calculo del coeficiente de correlacion de rango difuso."""
# T-norma de Lukasiewicz.
def T_norm(x,y):
    return max(0,x+y-1)

# TL-E-Ordenacion fuertemente completa en R.  
def Fuzzy_ordering(x,y,r):
    return min(1,max(0,1-((x-y)/r)))

# Relacion difusa estricta (Rx o Ry)
def Fuzzy_relation(x1,x2,r):
    return 1-Fuzzy_ordering(x2,x1,r)

# El grado de concordancia entre dos pares dado un r se calculara entonces de la siguiente forma.
def Concordance_degree(pair1,pair2,r):
    return T_norm(Fuzzy_relation(pair1[0], pair1[1],r), Fuzzy_relation(pair2[0], pair2[1],r))

""" Generacion de la matriz de grados de concordancia
Esta funcion devuelve la matriz de grados de concordancia del itemset de tamanio k=2 que se le pase como argumento,
junto con un r y el conjunto de datos."""
                  
# Utiliza float de 16 bits en lugar del float estandar en python que utiliza 64 bits.
def generate_matrix(candidate, df, r):
    
    
    at1 = pd.to_numeric(df.value[candidate[0][0]])
    op1 = candidate[0][1]
    
    at2 = pd.to_numeric(df.value[candidate[1][0]])
    op2 = candidate[1][1]
    
    N = len(at1)     # N es el numero de transacciones
    matrix = np.zeros((N, N), dtype = np.float16)
    
    if op1 == '>' and op2 == '>': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = Concordance_degree([at1[i], at1[j]],[at2[i], at2[j]],r)
                    
    if op1 == '>' and op2 == '<': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = Concordance_degree([at1[i], at1[j]],[at2[j], at2[i]],r)
                    
    return (tuple(candidate), csr(matrix))
                
"""
    Estas otras combinaciones nos las evaluaremos ya que solo habra itemsets de tamanio 2 del 
    tipo (('atributo1', '>'), ('atributo2', '>')) y (('atributo1', '>'), ('atributo2', '<'))

    if op1 == '<' and op2 == '>': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = Concordance_degree([at1[j], at1[i]],[at2[i], at2[j]],r)          
    
    

    if op1 == '<' and op2 == '<': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = Concordance_degree([at1[j], at1[i]],[at2[j], at2[i]],r)
"""                 
    
"""Generacion de la matriz de grados de concordancia. 8 bits
Variante de la funcion anterior para el caso en el que se elija una precision de 8bits, se utilizan enteros de 8 bits."""

# Utiliza enteros sin signo de 8 bits en lugar del float estandar en python que utiliza 64 bits.
def generate_matrix_int(candidate, df, r):
    
    
    at1 = pd.to_numeric(df.value[candidate[0][0]])
    op1 = candidate[0][1]
    
    at2 = pd.to_numeric(df.value[candidate[1][0]])
    op2 = candidate[1][1]
    
    N = len(at1)     # N es el numero de transacciones
    matrix = np.zeros((N, N), dtype = np.uint)
    
    if op1 == '>' and op2 == '>': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = (255*Concordance_degree([at1[i], at1[j]],[at2[i], at2[j]],r)) 
                    
    if op1 == '>' and op2 == '<': 
        for i in range(N):
            for j in range(N):
                if i != j:
                    matrix[i,j] = (255*Concordance_degree([at1[i], at1[j]],[at2[j], at2[i]],r))
                    
    return (tuple(candidate), csr(matrix))

"""Calculo del soporte
Devuelve el soporte de la matriz que se le pasa como argumento como la suma de todos sus elementos dividido entre N(N-1)/2"""

# Devuelve el soporte de la matriz
def compute_support(m):
    N = m.shape[1]
    return csr.sum(m)/(N*((N-1)/2))   # Tambien es posible usar np.sum directamente.


"""Calculo del soporte para 8 bits.
Variante de la funcion anterior para el caso en el que se elija una precision de 8bits."""

# Devuelve el soporte de la matriz
def compute_support_int(m):
    N = m.shape[1]
    return csr.sum(m/255)/(N*((N-1)/2))


"""Validacion del candidato combinado

Funcion que devuelve la combinacion de los itemsets pasados como argumento si esta combinacion es viable teniendo en cuenta los itemsets frecuentes de la iteracion anterior, evaluando si:

* Tienen al menos k-2 elementos en comun
* Todos sus subconjuntos de itemset graduales de tamanio k-1 fueron frecuentes en la iteracion o fase anterior.

La condicion de si ya han sido generados se evalua despues de esta funcion con un distinct() de la lista obtenida."""


def combinedCandidates(itemset_pair, previous_frequent_itemsets):
    #previous_frequent_itemsets = [set(i) for i in previous_frequent_itemsetsk]
    result = ()
    difference = set(itemset_pair[1])-set(itemset_pair[0])
    if(len(difference) == 1):
        result =  itemset_pair[0] + tuple(difference)
        combinations = [x for x in itertools.combinations(result, len(result)-1)]
        if not all(i in previous_frequent_itemsets for i in combinations):
            result = ()

    return result

"""Combinacion de matrices
Funcion que recibe como argumento a un itemset combinado y la lista de itemsets frecuentes de la iteracion anterior, con 
sus respectivas matrices. Devuelve el itemset y la matriz resultante de la combinacion de las matrices de dos de los itemsets
de tamanio k-1 que lo conforman, quedandonos con el minimo de cada uno de sus elementos (T-norma de Gödel de las matrices)."""

def ComputeCombinedMatrix(itemset, previous_frequent):
    
    itemset1 = itemset[:-1]
    itemset2 = itemset[1:]
    #itemset2 = itemset[:-2]+itemset[-1]
    
    
    matriz_resultante = csr.minimum(previous_frequent.value[itemset1], previous_frequent.value[itemset2])
    itemset_resultante = itemset1 + tuple(set(itemset2)-set(itemset1))
    
    return (itemset_resultante, matriz_resultante)



"""Funcion principal del algoritmo.
Devuelve la lista de todos los itemsets frecuentes en el conjunto de datos dado, que superen un cierto soporte minimo.
Recibe como entrada (argumentos) el SparkContext, el conjunto de datos(distribuido en el cluster), el soporte minimo,
un r para realizar las llamadas al calculo de grado de correlacion entre atributos. (Usa una precision de 16 bits para 
almacenar los grados de concordancia)"""

def extractFrequentItemsets(sc, distDataset, min_supp, r):
    
    # Primera fase del algoritmo
    candidates = []

    n_att = len(distDataset.value.columns)

    # Aniadimos a candidates todas los posibles itemsets de tamanio 2 (exceptuando equivalentes)
    for i in range(n_att):
        for j in range(i+1,n_att):
            candidates += [[(distDataset.value.columns[i],'>'),(distDataset.value.columns[j],'>')]]
            candidates += [[(distDataset.value.columns[i],'>'),(distDataset.value.columns[j],'<')]]

    
    # Evaluamos cada uno de los posibles candidatos de forma distribuida en el cluster, quedandonos unicamente
    # con los itemsets frecuentes de tamanio 2 junto con su matriz asociada.
    frequent_itemsets_k = sc.parallelize(candidates).map(lambda x: generate_matrix(x, distDataset, r))\
                                                    .filter(lambda x: compute_support(x[1]) >= min_supp)\
                                                    .collect()
    

    # Segunda fase del algoritmo

    frequent_itemsets_list = []
    # Repetimos este proceso hasta que la lista de itemsets frecuentes generados en la anterior iteracion esté vacia
    while(frequent_itemsets_k):
        # Creamos una copia de los itemsets frecuentes de la fase o iteracion anterior y sus respectivas matrices
        # en cada uno de los nodos del cluster.
        previous_frequent_dict = sc.broadcast(dict(frequent_itemsets_k))
        previous_frequent_itemsets = [item for item in previous_frequent_dict.value] # Lista con los itemsets frecuentes anterior
        frequent_itemsets_list += previous_frequent_itemsets
        candidates_combinations = [x for x in itertools.combinations(previous_frequent_itemsets, 2)]

        # Evaluamos las posibles combinaciones de candidatos de forma distribuida en el cluster, quedandonos unicamente
        # con los viables cuya matriz supere el soporte minimo.
        frequent_itemsets_k = sc.parallelize(candidates_combinations)\
                             .map(lambda x: combinedCandidates(x,previous_frequent_itemsets))\
                             .filter(lambda x: x)\
                             .distinct()\
                             .map(lambda x: ComputeCombinedMatrix(x, previous_frequent_dict))\
                             .filter(lambda x: compute_support(x[1]) >= min_supp)\
                             .collect()

    
    # Eliminamos el contexto actual.
    sc.stop()
    return frequent_itemsets_list

"""Funcion principal igual que la anterior pero para una precision de 8 bits."""

def extractFrequentItemsets_int(sc, distDataset, min_supp, r):
    
    # Primera fase del algoritmo
    candidates = []

    n_att = len(distDataset.value.columns)

    # Aniadimos a candidates todas los posibles itemsets de tamanio 2 (exceptuando equivalentes)
    for i in range(n_att):
        for j in range(i+1,n_att):
            candidates += [[(distDataset.value.columns[i],'>'),(distDataset.value.columns[j],'>')]]
            candidates += [[(distDataset.value.columns[i],'>'),(distDataset.value.columns[j],'<')]]

    # Evaluamos cada uno de los posibles candidatos de forma distribuida en el cluster, quedandonos unicamente
    # con los itemsets frecuentes de tamanio 2 junto con su matriz asociada.
    frequent_itemsets_k = sc.parallelize(candidates).map(lambda x: generate_matrix_int(x, distDataset, r))\
                                                    .filter(lambda x: compute_support_int(x[1]) >= min_supp)\
                                                    .collect()

    

    # Segunda fase del algoritmo

    frequent_itemsets_list = []
    # Repetimos este proceso hasta que la lista de itemsets frecuentes generados en la anterior iteracion esté vacia
    while(frequent_itemsets_k):
        # Creamos una copia de los itemsets frecuentes de la fase o iteracion anterior y sus respectivas matrices
        # en cada uno de los nodos del cluster.
        previous_frequent_dict = sc.broadcast(dict(frequent_itemsets_k))
        previous_frequent_itemsets = [item for item in previous_frequent_dict.value] # Lista con los itemsets frecuentes anterior
        frequent_itemsets_list += previous_frequent_itemsets
        candidates_combinations = [x for x in itertools.combinations(previous_frequent_itemsets, 2)]

        # Evaluamos las posibles combinaciones de candidatos de forma distribuida en el cluster, quedandonos unicamente
        # con los viables cuya matriz supere el soporte minimo.
        frequent_itemsets_k = sc.parallelize(candidates_combinations)\
                             .map(lambda x: combinedCandidates(x,previous_frequent_itemsets))\
                             .filter(lambda x: x)\
                             .distinct()\
                             .map(lambda x: ComputeCombinedMatrix(x, previous_frequent_dict))\
                             .filter(lambda x: compute_support_int(x[1]) >= min_supp)\
                             .collect()

    # Eliminamos el contexto actual.
    sc.stop()
    return frequent_itemsets_list

"""Inicializamos el SparkContext especificando el nombre de la aplicacion y el numero de nucleos 
(en caso de realizar una ejecucion en maquina local)"""

# Inicializacion del contexto
def createSparkContext():
    sc_conf = spark.SparkConf()
    sc_conf.setMaster("local[*]")    # Aqui podemos especificar el numero de nucleos del procesador que queremos que se utilicen en local
    sc_conf.setAppName("pattern mining")
    #sc_conf.set('spark.executor.instances', '2')
    #sc_conf.set('spark.executor.memory', '2g')
    #sc_conf.set('spark.executor.cores', '1')
    #sc_conf.set('spark.cores.max', '1')
    #sc_conf.set('spark.logConf', True)

    sc = spark.SparkContext(conf=sc_conf)

    return sc 


## Ejecucion mediante paso de argumentos por linea de comandos.
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()

    
    
    parser.add_argument('-f','--entrada', action='store', dest='dataset', default = "winequality-red.csv",
                    help='Fichero csv de datos de entrada')
    
    parser.add_argument('--separador', action='store', dest='separator', default = ";",
                    help='Separador usado en los datos de entrada')
    
    parser.add_argument('--n_att','-at', action='store', dest='num_atributos', default = 12,
                    help='Subconjunto de atributos que analizar de la base de datos')
    
    parser.add_argument('--n_trans','-tr', action='store', dest='num_transaciones', default = 100,
                    help='Subconjunto de transacciones que analizar de la base de datos')
    
    parser.add_argument('--soporte','-s', action='store', dest='soporte_minimo', default = 0.3,
                    help='Soporte minimo que debe superar un itemset para ser considerado frecuente')
    
    parser.add_argument('-r', action='store', dest='valor_r', default = 0.098,
                    help='Valor de r para el calculo de los grados de concordancia')
    
    parser.add_argument('-p', action='store', dest='precision', default = '16bits', choices = ['8bits','16bits'],
                    help='Precision usada para la representacion de los grados de concordancia')
    


    results = parser.parse_args()
    
    
    dataset_file = results.dataset
    sep = results.separator
    
    n_att = results.num_atributos
    n_trans = results.num_transaciones
    
    min_supp = results.soporte_minimo
    r = results.valor_r
    
    precision = results.precision
    
    
    

    sc = createSparkContext()

    # Ejecucion
    
    
    
    if(dataset_file == 'pd_speech_features.csv'):
        df = pd.read_csv("pd_speech_features.csv",sep = ",",skiprows = 1)   # Lo ponemos aparte porque tiene una fila extra y dos columnas que debemos quitar
        my_data = df.iloc[:n_trans,2:n_att+2]
    else:
    # Cargomos los datos en un dataframe de pandas.
    #df = pd.read_csv("seizure.csv",sep = ",")
        df = pd.read_csv(dataset_file,sep = sep)
    
    
    
    
    
    # Nos quedamos con un subconjunto total de las transacciones y de los atributos para adaptarlo a la capacidad de nuestro hardware
    #my_data = df.iloc[:80,1:-149]
    my_data = df.iloc[:n_trans,:n_att]
    #my_data = df.iloc[:100,1:-743]
    
    
    # Normalizo los datos
    normalized_data = (my_data - my_data.min())/(my_data.max() - my_data.min())
    
    datadist = sc.broadcast(normalized_data)
    
    if precision == '8bits':
        t1 = time.time()
    
        result = extractFrequentItemsets_int(sc,datadist, min_supp,r)
    
        t2 = time.time()
    else:  
        t1 = time.time()
        
        result = extractFrequentItemsets(sc,datadist, min_supp,r)
        
        t2 = time.time()
    
    
    
    f= open("resultados_ejecucion.txt","w+")
    f.write("Resultados obtenidos\n")
    f.write("Fichero de datos utilizado:\t"+dataset_file+"\n")
    f.write("Numero de atributos:"+repr(n_att)+"\t\t")
    f.write("Numero de transacciones:"+repr(n_trans)+"\n")
    f.write("Soporte minimo:"+repr(min_supp)+"\t\t")
    f.write("r:"+repr(r)+"\t\t")
    f.write("precision: "+precision+"\n")
    f.write("Tiempo de ejecucion:\t"+repr(t2-t1)+" Segundos\n\n\n")
    f.write("Itemsets frecuentes obtenidos:\n\n")
    for i in result:
        f.write(repr(i)+"\n")
    f.close()
    
    print("Tiempo de ejecucion del algoritmo usando Pyspark: ", t2-t1)
    print("Resultados guardados en resultados_ejecucion.txt")
       


