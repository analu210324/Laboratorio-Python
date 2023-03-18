#!/usr/bin/env python
# coding: utf-8

# #UNIVERSIDAD GALILEO
# ##Análisis de Datos con Python

# ## Parte 1: Numpy

# ### Ejercicio 1
# - ¿Cuál es el tamaño en bytes de un array de 1000 elementos de tipo booleano

# In[11]:


#importar la librería numpy

import numpy as np

#crear un arreglo de tipo booleano

arr = np.zeros(1000, dtype=bool)

# se calcula el tamaño del arreglo en bytes multiplicando el tamaño de cada elemento por el número de elementos en el arreglo

size = arr.itemsize * arr.size

#se imprime resultado
print(f"El tamaño en bytes de un array de 1000 elementos de tipo booleano es: {size} bytes")


# ### Ejercicio 2
# - Cree un array cuyos elementos sean los enteros pares en [1,100] y en orden decreciente. Muestre los 10 últimos por pantalla.

# In[13]:


#se importa la librería numpy

import numpy as np

# se crea un arreglo del 2 al 100, tomando sólo los números pares.
arr = np.arange(2, 101, 2)

#se invierte el arreglo oara crear el orden decreciente

arr = np.flip(arr)

#del arreglo invertido se imprime los últimos 10 enteros pares 
print(arr[-10:])


# ### Ejercicio 3 Dado el array
# 2 4 5 6
# 0 3 7 4
# 8 8 5 2
# 1 5 6 1
# Seleccione con una instrucción el subarray de elementos
# 0 3 7 4
# Después, seleccione el subarray de elementos
# 2 5
# 8 5

# In[21]:


#importamos la librería numpy
import numpy as np

#Se crea una matriz de 4x4 con valores específicos.
arr = np.array([[2, 4, 5, 6], [0, 3, 7, 4], [8, 8, 5, 2], [1, 5, 6, 1]])

#se crea un arreglo se toma una submatriz de la matriz original, se específica la fila 1 y las columnas de 0 a 4
subarr1 = arr[1, 0:4]
#se imprime la submatriz 1
print(subarr1)

# se crea otra submatriz a partir de la matriz original, se seleccionan las filas 0 y 2, y las columnas 0 y 2
subarr2 = arr[[0, 2]][:, [0, 2]]

#se imprime la submatriz 2

print(subarr2)


# ### Ejercicio 4  Dados los arrays a = [1, 4, 2, 7] y b = [1, 3, 2, 9], obtenga la media aritmética de la diferencia (a-b).
# 

# In[22]:


#se importa la librería numpy
import numpy as np

#se definen los arreglos a y b con los valores específicos
a = np.array([1, 4, 2, 7])
b = np.array([1, 3, 2, 9])

#se crea una diferencia entre los arreglos a y b
diff = a - b

#se calcula el valor medio(media aritmetica) de la diferencia antes creada en la variable diff
mean_diff = np.mean(diff)

#se imprime el resultado de la media aritmética
print(mean_diff)


# ### Ejercicio 5 Tengo valores de cordenadas (x, y) en las columnas del array
# 1.33 4.5
# 30.0 10.7
# 70.2 0.5
# Agregue a este array las coordenadas (37.1, -3.6). Muestre en pantalla las dimensiones del nuevo array

# In[24]:


#se importa la variable numpy

import numpy as np

#se crea una matriz bidimencional con tres filas y dos columnas

arr = np.array([[1.33, 4.5], [30.0, 10.7], [70.2, 0.5]])

#se crea un arreglo con las nuevas coordenadas
new_coord = np.array([37.1, -3.6])

# el arreglo que con nuevas coordenadas se alamacena en la variable arr con la funcion vstack que concatena las coordenadas
arr = np.vstack((arr, new_coord))

#se muestra el resultado de las nuevas dimensiones del arreglo: 4 filas 2, columnas
print(arr.shape)


# In[ ]:


###Ejercicio 6
-- Copie el array del ejercicio anterior. Traspóngalo (Agregue ahora dos nuevos pares de coordenadas: (10.8, 3.0) y (35.8, 12.0)


# In[30]:


#se importa la librería numpy
import numpy as np

# se crea una matriz con las coordenadas específicas 

arr = np.array([[1.33, 4.5], [30.0, 10.7], [70.2, 0.5]])

#se crea un nuevo arreglo con las nuevas coordenadas

new_coords = np.array([[10.8, 3.0], [35.8, 12.0]])

#se concatenan los valores almacenados entre la matriz y arreglo 

arr = np.vstack((arr, new_coords))

#se utiliza laf uncion transpose para transponer la matriz intercambiandose filas por columnas

arr_t = np.transpose(arr)

#se imprime el resultado 
print(arr)


# ## Parte 2: Exploración y Minería de Datos

# In[39]:


# se importa la librería pandas

import pandas as pd

# e carga el archivo Csv en un objeto de dataframe declarado como df y se lee el archivo con la funcion read_csv
df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

# se imprimen las primeras 5 filas del dataframe para conocer los nombres de las columnas y continuar con los cálculos
print(df.head())


# ### 1) ¿Cuántos Municipios tienen más de 100000 habitantes?.

# In[56]:


#se crea un dataframe municipios y se cuenta los municipios con el metodo nunique que tienen mayor poblacion a 100000 habitantes en una serie de datos

municipios = df[df['Poblacion'] > 100000]['Municipio'].nunique()

#se imprime los municipios contados en la serie.

print( municipios, 'municipios con más de 100,000 habitantes.')


# ### 2) Realice una gráfica de barras sobre la población de cada ciudad, ordenela de menor a mayor y responda:
# a. ¿Cuál es la segunda ciudad más poblada?
# b. ¿Qué posición ocupa Granada en el ranking de las ciudades más pobladas

# In[71]:


#se importa la librería pandas

import pandas as pd

#se importa la librería matplotlib.pylot para graficar

import matplotlib.pyplot as plt

# se lee el archivo NucleosPoblacion.csv

df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

#se ordenan la por poblacion de forma descendente

df = df.sort_values('Poblacion', ascending=False)

# se grafica con barras  tomando: X= municipios y Y=publación

plt.bar(df['Municipio'], df['Poblacion'])

# se crea un título a la gráfica(opcional)

plt.title('Población de cada ciudad')

# se crea un título para el eje X

plt.xlabel('Ciudad')

# se crea un título para el eje Y

plt.ylabel('Población')

# se giran en 90 grados las etiquetas (opcional)

plt.xticks(rotation=90)

#se muestra el resultado de la gráfica

plt.show()

#se calcula cual es la ciudad más poblada con tomando en cuenta que está ordenado en forma descendente, devuelve la fila que está en la posición -2 del DataFrame df

segunda_ciudad_mas_poblada = df.iloc[-2]['Municipio']

#se imprime el resultado del dataframe segunda_ciudad_mas_poblada

print("La segunda ciudad más poblada es:", segunda_ciudad_mas_poblada)

# se calcula que posición ocupa en el ranking Granada, en el dataFrame df c

granada_pos = df['Municipio'].tolist().index('Granada')

#asumimos que granada es una de las ciudades más pobladas, para encontrar el índice real de granada en la lista
ranking_pos = granada_pos + 1

#se imprime el resultado de variable anterior

print("La ciudad de Granada ocupa la posición número", ranking_pos, "en el ranking de las ciudades más pobladas.")


# ### 3. ¿Cuántos municipios de Extremadura tienen más de 5000 habitantes?

# In[72]:


#se importa la librerpia pandas

import pandas as pd

#se lee el documento NucleosPoblacion.csv

df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

#se crea un filtro para los municipios de Extremadura, que se encuentran en la columna CodProvin y tienen código 6

df_extremadura = df[df['CodProvin'] == 6]

#se filtra nuevamente de la variable anterior y se cuentan cuantos municipios en exremadura tienen la publacion  mayor a 5000 habitantes

num_municipios = len(df_extremadura[df_extremadura['Poblacion'] > 5000])

# se imprime el resultado 

print(num_municipios, "municipios en Extremadura tienen más de 5000 habitantes")


# ### 4. ¿Cuál es el municipio situado más al Norte? (Usar el valor de la coordenada "Y" que representa la latitud en grados). Proporcione también la provincia a la que pertenece y su población.

# In[73]:


#se importa la librería pandas
import pandas as pd

#se lee el archivo NucleosPoblacion.csv
df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

#se calcula el valor maximo de la columna Y del dataframe df

max_y = df['Y'].max()

#se filtran de la variable creada anteriormente sólo aquellos que tienen valor máximo

municipio_norte = df[df['Y'] == max_y]

# se almacena en la variables municipio, provincia, publacion  los que se encuentran en la variable anterior tomando la que tiene valor 0
# esta representa el municipio situado más al norte.
nombre_municipio = municipio_norte['Municipio'].iloc[0]
provincia = municipio_norte['Provincia'].iloc[0]
poblacion = municipio_norte['Poblacion'].iloc[0]

#se imprime el resultado:

print("El municipio más al norte es", nombre_municipio, "que pertenece a la provincia de", provincia, "y tiene una población de", poblacion, "habitantes.")


# ### 5. btenga la media, mediana, desviación estándar, valor máximo y valor mínimo de la población de los municipios de la provincia de Granada

# In[74]:


#se importa la librería pandas

import pandas as pd

#se lee el archivo csv
df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

# se filtra el municipio granada
municipios_granada = df[df['Provincia'] == 'Granada']

#con el método .describe se crean las estadísticas descriptivas básicas (la media, la mediana, la desviación estándar, el valor máximo y el valor mínimo)
estadisticas_poblacion = municipios_granada['Poblacion'].describe()

#se imprime la media, mediana, desviación, maximo y minimo basado en en método .describe
print("Media de población:", estadisticas_poblacion['mean'])
print("Mediana de población:", estadisticas_poblacion['50%'])
print("Desviación estándar de población:", estadisticas_poblacion['std'])
print("Valor máximo de población:", estadisticas_poblacion['max'])
print("Valor mínimo de población:", estadisticas_poblacion['min'])


# ### 6 Realice un histograma con la población de los Municipios para cada una de las provincias

# In[78]:


#se importa la librería pandas
import pandas as pd

#se utiliza el metodo .unique para devolver únicas provincias
provincias = df['Provincia'].unique()

#se crea un for para obtener los municipios de las provincia actual, se grafica con plt.hist  
# a la gráfica se le puede configurar transparencia, nombre de las label, intervalos, etc
for provincia in provincias:
    municipios_provincia = df[df['Provincia'] == provincia]
    plt.hist(municipios_provincia['Poblacion'], bins=25, alpha=0.7, label=provincia)
    
#se imprime un título (opcional)
plt.title('Histograma de población de municipios por provincia')
#se imprime un ejex   (opcional)
plt.xlabel('Población')
#se imprime un ejey   (opcional)
plt.ylabel('Frecuencia')
#se imprime leyenda    (opcional)
plt.legend()
# se muestra gráfico
plt.show()


# ### 7. Seleccione al azar cincuenta municipios diferentes de entre los diponibles en el archivo. Asegúrese de que no se repitan. ¿Luego calcule el promedio de la población y la desviación estándar de esto 50 municipios?.

# In[81]:


#se importa librería pandas y se lee archivo .csv
import pandas as pd
df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

# se crea una variable que contenga  que tome 50 valores de forma aleatoria del dataframe df y que estos no sean repetidos 
df_sample = df.sample(n=50, replace=False, random_state=1)

# se crea la variable que contendrá la funcion del promedio de la publacion con el metodo .mean
poblacion_mean = df_sample['Poblacion'].mean()

# se crea la variable que contendrá la funcion de la desviacione standar con el metodo .std

poblacion_std = df_sample['Poblacion'].std()

#se imprime cada uno de los resultados:
# La expresión {:.2f} se utiliza para formatear un número en una cadena de texto lo que indica que se desea formatear como un número de punto flotante con dos dígitos después del punto decimal
print("El promedio de población de los 50 municipios seleccionados: {:.2f}".format(poblacion_mean))
print("La desviación estándar de la población de los 50 municipios seleccionados: {:.2f}".format(poblacion_std))


# ### 8. Dígame los nombres de los Municipios más cercano y más lejano a Madrid. Para ello debe calcular la distancia en todos ellos y Madrid. No considere a Madrid en el análisis ya que la distancia sería cero

# In[88]:


#se importa las librerpias pandas y random
import pandas as pd
import random

# se importa la funcion geodesic, si no se cuenta debe instalarse desde el cmd: pip install geopy

from geopy.distance import geodesic

#se lee el archivo .csv
df = pd.read_csv(r'C:\PY\NucleosPoblacion.csv')

#tomar de google maps para definir latitud y longitud pues este es 0 a un inicio
madrid_latitud = 40.416775
madrid_longitud = -3.703790

#se crea una funcion que cacule la distancia entre latitud y longitud entre municipios y que devuelva los valores en km usando la libreria geodesic
def distancia(lat1, long1, lat2, long2):
    return geodesic((lat1, long1), (lat2, long2)).km

df['distancia_madrid'] = df.apply(lambda row: distancia(row['Y'], row['X'], madrid_latitud, madrid_longitud), axis=1)

# se excluye madrid del dataframe pues tenía valor 0 y afecta el cálculo

df = df[df['Municipio'] != 'MADRID']

#Se alamacena le municipio  más cercano a madrid basado en le valór minimo almacenado en municipio cercano

municipio_cercano = df.loc[df['distancia_madrid'].idxmin(), 'Municipio']

#se almacena la distancia existente 

distancia_cercana = df['distancia_madrid'].min()


#Se alamacena le municipio  más lejana a madrid basado en le valór máximo almacenado en municipio lejano
municipio_lejano = df.loc[df['distancia_madrid'].idxmax(), 'Municipio']

#se almacena la distancia existente 

distancia_lejana = df['distancia_madrid'].max()

#se imprimen resultados
print(f"El municipio más cercano a Madrid es {municipio_cercano} a una distancia de {distancia_cercana} km.")
print(f"El municipio más lejano a Madrid es {municipio_lejano} a una distancia de {distancia_lejana} km.")


# In[ ]:




