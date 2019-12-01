# -*- coding: utf-8 -*-
"""
@author: Antonio Javier
"""
import numpy as np
import pandas as pd
import time
import itertools
import argparse



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
Esta funcion devuelve  la matriz de grados de concordancia de los dos items graduales g_item1 y g_item2 que se le 
pasan como argumento, junto con un r y el conjunto de datos."""
                  

def generate_matrix(df, r, g_item1, g_item2):
    at1 = pd.to_numeric(df[g_item1[0]])
    op1 = g_item1[1]
    
    at2 = pd.to_numeric(df[g_item2[0]])
    op2 = g_item2[1]
    
    N = len(at1)
    matrix = np.zeros((N, N))
    
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
                    
    return matrix
                
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

                    


"""Calculo del soporte
Devuelve el soporte de la matriz que se le pasa como argumento como la suma de todos sus elementos dividido entre N(N-1)/2"""

def compute_support(m):
    N = len(m[0])
    return np.sum(m)/(N*((N-1)/2))



"""Validacion del candidato combinado
Función que combina dos itemsets en caso de que tengan al menos k-2 elementos en común, en caso contrario devuelve 0
"""
def combineCandidates(itemset1, itemset2, candidates):
    # Primero comparamos que tengan al menos k-2 en comun
    result = 0
    difference = set(itemset2)-set(itemset1)
    if(len(difference) == 1):
        result =  itemset1 + list(difference)
        combinations = [x for x in itertools.combinations(result, len(result)-1)]
        if not all(list(i) in candidates for i in combinations): 
            result = 0

    return result



"""Funcion principal del algoritmo.
Funcion que devuelve los itemsets frecuentes en el conjunto de datos ''Dataset'' teniendo en cuenta un umbral de soporte
de ''min_supp'' y un r dado en el parametro ''r''."""

def extractFrequentItemsets(dataset, min_supp, r):
    
    """ Primera parte del algoritmo. Generar los candidatos frecuentes de k = 2 con sus respectivas matrices """
    candidates = []
    
    n_att = len(dataset.columns)
    
    for i in range(n_att):
        for j in range(i+1,n_att):
            candidates += [[(dataset.columns[i],'>'),(dataset.columns[j],'>')]]
            candidates += [[(dataset.columns[i],'>'),(dataset.columns[j],'<')]]

        
    
    
    frequent_matrices = []
    
    frequent_candidates = []
    for itemset in candidates:
        candidate_matrix = generate_matrix(dataset, r, itemset[0], itemset[1])
        candidate_support = compute_support(candidate_matrix)
        if(candidate_support >= min_supp):
            frequent_matrices += [candidate_matrix]
            frequent_candidates += [itemset]


    """
    Segunda parte del algoritmo. Generar los candidatos frecuentes de k > 2 con sus respectivas matrices
    """
       
    frequent_patterns = []
    
    generated_candidates = []
    
    while(frequent_candidates):
        frequent_patterns += frequent_candidates
        
        candidates = frequent_candidates
        
        matrices = frequent_matrices
        
        frequent_matrices = []
        
        frequent_candidates = []
        
        num_candidates = len(candidates)
        
       
        
        for i in range(num_candidates):
            for j in range(i+1,num_candidates):
                current_candidate = combineCandidates(candidates[i], candidates[j], candidates)        
                already_generated = 0
                if (current_candidate != 0):
                    for cand in generated_candidates:
                        already_generated += all(elem in cand for elem in current_candidate)
                    if(already_generated == 0):
                        generated_candidates += [current_candidate]
                        candidate_matrix = np.minimum(matrices[i],matrices[j]) 
                        candidate_support = compute_support(candidate_matrix)
                        if(candidate_support >= min_supp):
                            frequent_matrices += [candidate_matrix]
                            frequent_candidates += [current_candidate]

    return frequent_patterns
        


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
    
    


    results = parser.parse_args()
    
    
    dataset_file = results.dataset
    sep = results.separator
    
    n_att = results.num_atributos
    n_trans = results.num_transaciones
    
    min_supp = results.soporte_minimo
    r = results.valor_r
    
    
    

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
    
    
    # Normalizamos los datos
    normalized_data = (my_data - my_data.min())/(my_data.max() - my_data.min())
    


    t1 = time.time()
    
    result = extractFrequentItemsets(normalized_data, min_supp,r)
    
    t2 = time.time()

    
    
    f= open("resultados_ejecucion.txt","w+")
    f.write("Resultados obtenidos\n")
    f.write("Fichero de datos utilizado:\t"+dataset_file+"\n")
    f.write("Numero de atributos:"+repr(n_att)+"\t\t")
    f.write("Numero de transacciones:"+repr(n_trans)+"\n")
    f.write("Soporte minimo:"+repr(min_supp)+"\t\t")
    f.write("r:"+repr(r)+"\t\t")
    f.write("Tiempo de ejecucion:\t"+repr(t2-t1)+" Segundos\n\n\n")
    f.write("Itemsets frecuentes obtenidos:\n\n")
    for i in result:
        f.write(repr(i)+"\n")
    f.close()
    
    print("Tiempo de ejecucion del algoritmo base: ", t2-t1)
    print("Resultados guardados en resultados_ejecucion.txt")
       


