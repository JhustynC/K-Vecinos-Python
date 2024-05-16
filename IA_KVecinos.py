from collections import Counter
from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as dist
from tkinter import filedialog
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

dataframe = pd.DataFrame()
root = tk.Tk

def lecturaArchivo():

    global dataframe
    archivo_csv = filedialog.askopenfilename(filetypes=[("Archivo CSV", "*.csv")])
    # Leer el archivo CSV seleccionado
    try:
        dataframe.drop(index=dataframe.index, columns=dataframe.columns, inplace=True)
        datos = pd.read_csv(archivo_csv)
        dataframe = pd.concat([dataframe, datos], ignore_index=True)
        print("Datos ingresados correctamente")
        root.destroy()
        return dataframe
    except FileNotFoundError:
        print("No se eligió el archivo")

def calcularKVecinos(mejorK):
    num_filas, num_columnas = dataframe.shape
    print("\nNumero de variables: ", num_columnas-1)
    nombres_columnas = list(dataframe.columns)
    indice = int(0.8 * dataframe.shape[0])  # calcula el indice del 80% de las filas
    print("\nSe toma el 80% de los datos, que son: ", indice)
    nombre_ultima_columna = dataframe.columns[-1]  # obtiene el nombre de la ultima columna
    datosEntrenamiento = dataframe.iloc[:indice]    # selecciona solo las filas hasta ese indice
    datosPrueba = dataframe.iloc[indice:]  # selecciona solo las filas desde ese indice
    print("\n=======DatosEntrenamiento=======")
    print(datosEntrenamiento)
    print("\n=======DatosPrueba=======")
    print(datosPrueba)
    palette = sns.color_palette("bright", len(datosEntrenamiento[nombre_ultima_columna].unique()))
    color_dict = dict(zip(datosEntrenamiento[nombre_ultima_columna].unique(), palette))
    frecuencias = dataframe.iloc[:indice, -1].value_counts().reset_index()
    frecuencias.columns = ['Clase', 'Frecuencia']
    if num_columnas-1 <3:
        dibujar2D(nombres_columnas,datosEntrenamiento,color_dict,nombre_ultima_columna)
    elif num_columnas-1 == 3:
        dibujar3D(nombres_columnas,datosEntrenamiento,nombre_ultima_columna)
    print(frecuencias)
    distancia = []
    listaTabla = []
    for i in range(indice, dataframe.shape[0]):
        for j in range(0,indice):
            distancia.append(dist.euclidean(dataframe.iloc[i, :-1].values, dataframe.iloc[j, :-1].values))
        dfModificado = datosEntrenamiento.copy()
        dfModificado['DistanciasCalculadas'] = distancia
        dfModificado.sort_values(by=['DistanciasCalculadas'], inplace=True)
        print("")
        print(dfModificado)
        listaTabla.append(dfModificado.copy())
        dfModificado.drop(index=dfModificado.index, columns=dfModificado.columns, inplace=True)
        distancia.clear()
    bandera=True
    n=0
    aux=0
    listaClases = []
    dicK = {}
    while True:
        for i in listaTabla:
            k=1
            print("")
            print("Dato de prueba:")
            print(datosPrueba.iloc[aux,:-1].to_frame().T.to_string(index=False))
            print("")
            for j in range(0,listaTabla[0].shape[0]):
                listaClases.append(i.iloc[j,-2])
                if n >= len(listaTabla):
                    bandera=False
                print("K = ",k)
                print(listaClases)
                freq = Counter(listaClases)
                print('{:5s} {:10s}'.format('Clases','Frecuencia'))
                for numero, frecuencia in freq.items():
                    print('{:5d} {:10d}'.format(numero,frecuencia))
                numero_mas_comun = max(freq, key=freq.get)
                print("")
                print("La clase predicha para este dato de prueba es: ",numero_mas_comun)
                dicK.setdefault(f"K = {k}", []).append(numero_mas_comun)
                k+=1
                n+=1
            listaClases.clear()
            aux+=1
        if bandera==False:
            break

    print(dicK)
    contadorDeK =0
    precisionMax =float('-inf')
    for clave, valores in dicK.items():
        cont =0
        var =0
        contadorDeK+=1
        for valor in valores:
            if valor == datosPrueba.iloc[var,-1]:
                cont+=1
            var+=1
        precision = (cont /len(datosPrueba))*100
        if(precisionMax<precision):
            mejorK = contadorDeK
            precisionMax = precision

        print(f"k = {contadorDeK} tiene una ",precision,"%")
    print(f"El mejor k es {mejorK} con un ", precisionMax)
    return mejorK

def dibujar2D(nombres_columnas,sub_df,color_dict,nombre_ultima_columna):
    sns.scatterplot(x=nombres_columnas[0], y=nombres_columnas[1], hue=nombre_ultima_columna, data=sub_df, palette=color_dict)
    plt.scatter(sub_df.iloc[:,0], sub_df.iloc[:,1], c=sub_df[nombre_ultima_columna].apply(lambda x: color_dict[x]), marker='s')
    plt.xlabel(nombres_columnas[0])
    plt.ylabel(nombres_columnas[1])
    plt.show()

def dibujar3D(nombres_columnas, df, nombre_ultima_columna):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    clases = df[nombre_ultima_columna].unique()
    color_dict = {clase: np.random.rand(3,) for clase in clases} # generamos un diccionario con un color aleatorio por cada clase
    for clase, color in color_dict.items():
        temp_df = df[df[nombre_ultima_columna] == clase]
        ax.scatter(temp_df[nombres_columnas[0]], temp_df[nombres_columnas[1]], temp_df[nombres_columnas[2]], color=color, marker='s', label=str(clase))
    ax.set_xlabel(nombres_columnas[0])
    ax.set_ylabel(nombres_columnas[1])
    ax.set_zlabel(nombres_columnas[2])
    plt.legend()
    plt.show()

def predecirClasificacion(k):
    """print(k)
    if dataframe.empty:
        print("Primero debe cargar un archivo para entrenar el modelo")
        return
    if k==0:
        print("Primero debe entrenar el modelo")
        return
    dato = input("Ingrese los datos a predecir separados por comas: ")
    dato = dato.split(",")
    dato = [float(x) for x in dato]
    print(dato)
    distancia = []
    for i in range(dataframe.shape[0]):
        distancia.append(dist.euclidean(dato, dataframe.iloc[i, :-1]))
    df_modificado = dataframe.copy()
    df_modificado['DistanciasCalculadas'] = distancia
    print(df_modificado)
    df_modificado.sort_values(by=['DistanciasCalculadas'], inplace=True)
    print(df_modificado)
    clases = df_modificado.iloc[:k,-2].values
    print(clases)
    frecuencia_clases = Counter(clases)
    clase_predicha = max(frecuencia_clases, key=frecuencia_clases.get)
    print(f"La clase predicha es: {clase_predicha}")"""

    #CAMBIO PARA INTERCICLO
    print(k)
    if dataframe.empty:
        print("Primero debe cargar un archivo para entrenar el modelo")
        return
    if k == 0:
        print("Primero debe entrenar el modelo")
        return

    num_datos = int(input("Ingrese la cantidad de datos a predecir: "))
    datos = []
    for _ in range(num_datos):
        dato = input("Ingrese los datos a predecir separados por comas: ")
        dato = dato.split(",")
        dato = [float(x) for x in dato]
        datos.append(dato)
    valorK = int(input("Ingrese el valor de k: "))
    classPred = []
    for dato in datos:
        print(dato)
        distancia = []
        indice = int(0.8 * dataframe.shape[0])  # calcula el indice del 80% de las filas
        df_modificado = dataframe.iloc[:indice]    # selecciona solo las filas hasta ese indice
        for i in range(df_modificado.shape[0]):
            distancia.append(dist.euclidean(dato, df_modificado.iloc[i, :-1]))
        dfmod = df_modificado.copy()
        dfmod['DistanciasCalculadas'] = distancia
        print(dfmod)
        dfmod.sort_values(by=['DistanciasCalculadas'], inplace=True)
        print(dfmod)
        clases = dfmod.iloc[:valorK, -2].values
        print(clases)
        frecuencia_clases = Counter(clases)
        clase_predicha = max(frecuencia_clases, key=frecuencia_clases.get)
        print(f"La clase predicha es: {clase_predicha}")
        classPred.append(clase_predicha)
    print("Las clases predichas son:", classPred)

def menu():
    mejorK = 0
    while True:
        print("\n===============Algoritmo de K-Vecinos===============")
        print("1. Ingresar datos desde archivo .csv")
        print("2. Mostrar Datos")
        print("3. Calcular K-Vecinos")
        print("4. Predecir nuevo dato")
        print("5. Salir")
        try:
            opc = int(input("Opcion: "))
            if opc == 1:
                try:
                    global root
                    dataframe.drop(index=dataframe.index, columns=dataframe.columns, inplace=True)
                    # Crear la ventana principal
                    root = tk.Tk()
                    
                    # Obtener las dimensiones de la pantalla
                    ancho_pantalla = root.winfo_screenwidth()
                    alto_pantalla = root.winfo_screenheight()

                    # Definir las dimensiones de la ventana
                    ancho_ventana = 100
                    alto_ventana = 20

                    # Calcular la posición para centrar la ventana
                    posicion_x = (ancho_pantalla - ancho_ventana) // 2
                    posicion_y = (alto_pantalla - alto_ventana) // 2

                    # Establecer la geometría de la ventana para centrarla
                    root.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")
                    
                    root.lift()
                    root.focus_force()
                    
                    # Agregar un botón para abrir el cuadro de diálogo de selección de archivos
                    boton_abrir = tk.Button(root, text="Abrir archivo CSV", command=lecturaArchivo)
                    boton_abrir.pack()
                    
                    # Mostrar la ventana principal
                    root.mainloop()
                except:
                    print("No se eligió el archivo")
            elif opc == 2:
                print(dataframe)
            elif opc == 3:
                print('\n============ Calcular los K-Vecinos ============')
                #mejorK=calcularKVecinos(mejorK)
                # Separar las características (variables independientes) y la variable objetivo (clase)
                X = dataframe.drop('Clase', axis=1)
                y = dataframe['Clase']

                # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Normalizar las características
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Lista para almacenar la precisión media para cada valor de K
                mean_accuracy = []

                # Rango de valores de K que deseas probar
                k_values = range(1, 21)

                # Realizar la validación cruzada para cada valor de K
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                    mean_accuracy.append(scores.mean())

                # Encontrar el valor de K que maximiza la precisión media
                optimal_k = k_values[mean_accuracy.index(max(mean_accuracy))]
                print("El valor óptimo de K es:", optimal_k)

                # Graficar la precisión media en función del valor de K
                plt.plot(k_values, mean_accuracy)
                plt.xlabel('Valor de K')
                plt.ylabel('Precisión Media')
                plt.title('Precisión Media vs Valor de K')
                plt.show()
            elif opc == 4:
                predecirClasificacion(mejorK)
            elif opc == 5:
                print("Finalizando...")
                break
        except ValueError:
            print("Opcion invalida")

menu()