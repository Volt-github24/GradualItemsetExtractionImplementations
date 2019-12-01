# Intrucciones de Ejecución

A continuación se detallan las instrucciones de ejecución por línea de comandos para cada uno de los archivos .py, todos ellos devuelven la salida obtenida (el tiempo de ejecución, los parámetros elegidos y los itemsets frecuentes obtenidos) en un fichero “resultados\_ejecucion.txt”.
Cabe mencionar que para la ejecución de estos archivos en necesario tener instalada la versión de python 3.6 o superior.
Cada programa se puede ejecutar sin opciones ya que cuenta con los siguientes parámetros por defecto en caso de que no se especifique alguno de ellos:

1.  **Conjunto de datos:** winequality-red.csv

2.  **Separador:** ;

3.  **Número de atributos:** 12

4.  **Número de transacciones:** 100

5.  **Soporte Mínimo:** 0.3

6.  **r:** 0.098

7.  **Mejora:** m2 (exclusivo de las implementacion de mejoras por separado)

8.  **Precisión:** 16 bits (exclusivo de las implementaciones de Spark)

### AlgoritmoBase.py

Para introducir los distintos parámetros de entrada del algoritmo base tenemos las siguientes opciones:

``` {.bash language="bash"}
  $ python AlgoritmoBase.py [-h] [-f <conjunto de datos>] [--separador <separador>][--n_att <numero de atributos>] [--n_trans <numero de transacciones>][--soporte <soporte minimo>] [-r <valor de r>]
```

-   **-h, –help**: Mostrar mensaje de ayuda.

-   **-f , –entrada**: Fichero csv de datos de entrada.

-   **–separador**: Separador usado en los datos de entrada.

-   **–n\_att, -at**: Subconjunto de atributos que analizar de la base de datos.

-   **–n\_trans, -tr**: Subconjunto de transacciones que analizar de la base de datos

-   **–soporte , -s**: Soporte mínimo que debe superar un itemset para ser considerado frecuente

-   **-r**: Valor de r para el calculo de los grados de concordancia

### AlgoritmoMejorasSeparadas.py

Para introducir los distintos parámetros de entrada del algoritmo con las mejoras separadas tenemos las siguientes opciones:

``` {.bash language="bash"}
  $ python AlgoritmoMejorasSeparadas.py [-h] [-f <conjunto de datos>] [--separador <separador>][--n_att <numero de atributos>] [--n_trans <numero de transacciones>][--soporte <soporte minimo>] [-r <valor de r>] [--mejora {m1,m2}]
```

Igual que el anterior exceptuando a:
**–mejora**: Mejora usada: m1 para matrices comprimidas de Yale. Y m2 para mejora de precision con 8 bits

### AlgoritmoMejorado.py

Para introducir los distintos parámetros de entrada del algoritmo mejorado tenemos las siguientes opciones:

``` {.bash language="bash"}
  $ python AlgoritmoMejorado.py [-h] [-f <conjunto de datos>] [--separador <separador>][--n_att <numero de atributos>] [--n_trans <numero de transacciones>][--soporte <soporte minimo>] [-r <valor de r>]
```

Todas sus opciones ya han sido descritas.

### ImplementacionSpark.py

Para introducir los distintos parámetros de entrada de la implementacion en Spark tenemos las siguientes opciones:

``` {.bash language="bash"}
  $ python ImplementacionSpark.py [-h] [-f <conjunto de datos>] [--separador <separador>][--n_att <numero de atributos>] [--n_trans <numero de transacciones>][--soporte <soporte minimo>] [-r <valor de r>][-p <precision>{8bits,16bits}]
```

La única opción que contiene que no hemos introducido todavía es:
**-p**: Precisión usada para la representación de los grados de concordancia

### ImplementacionSpark.ipynb

Este archivo contiene el mismo código que “ImplementacionSpark.py” (exceptuando las líneas de código referentes a las opciones por línea de comandos). Se trata del mismo programa para ejecutar función por función en Jupyter Notebook donde se puede experimentar más cómodamente con los parámetros del contexto de Spark (SparkContext).
