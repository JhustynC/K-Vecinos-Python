import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Inicializar el objeto StandardScaler
scaler = StandardScaler()

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points with n-dimensions.
    """
    return np.sqrt(np.sum((point1 - point2)**2))

def k_nearest_neighbors(df, k, reference_index):
    """
    Calculate the k-nearest neighbors for a given reference point in a DataFrame.
    """
    distances = []
    reference_point = df.iloc[reference_index, :-1]  # Exclude the last column (class)

    # Calculate distances from reference point to all other points
    for i, row in df.iterrows():
        if i != reference_index:
            other_point = row[:-1]  # Exclude the last column (class)
            distance = euclidean_distance(reference_point, other_point)
            distances.append((i, distance, row.iloc[-1]))  # Include class value

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])

    # Get the k-nearest neighbors
    k_nearest = distances[:k]
    
    return k_nearest

def neighbors_predict_class(df, k, new_point):
    """
    Calculate the k-nearest neighbors for a given new point in a DataFrame.
    Returns the predicted class for the new point, along with distances and neighbor indices.
    """
    distances = []

    # Calculate distances from new point to all other points
    for i, row in df.iterrows():
        point = row[:-1]  # Exclude the last column (class)
        distance = euclidean_distance(new_point, point)
        distances.append((i, distance, row.iloc[-1]))  # Include class value

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])

    # Get the k-nearest neighbors
    k_nearest = distances[:k]

    # Count occurrences of each class in the k-nearest neighbors
    class_count = {}
    for _, dist, neighbor_class in k_nearest:
        class_count[neighbor_class] = class_count.get(neighbor_class, 0) + 1 / dist  # Weighted by distance

    # Predict the class of the new point based on the majority class among the k-nearest neighbors
    predicted_class = max(class_count, key=class_count.get)
    
    return predicted_class, k_nearest


# Load data from CSV into a DataFrame
csv_file = 'kvecinos3.csv'  # Reemplaza 'datos.csv' con el nombre de tu archivo CSV
data = pd.read_csv(csv_file)

# # Seleccionar solo las columnas numéricas para la normalización
# numeric_columns = ['Variable1', 'Variable2', 'Variable3']
# numeric_data = data[numeric_columns]

# # Inicializar el objeto MinMaxScaler
# scaler = MinMaxScaler()

# # Normalizar los datos numéricos
# normalized_data = scaler.fit_transform(numeric_data)

# # Convertir los datos normalizados de vuelta a DataFrame
# normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

# # Agregar la columna 'Clase' al DataFrame normalizado
# normalized_df['Clase'] = data['Clase']

# print('\nMostrar el DataFrame original')
# print(data)

# print('\nMostrar el DataFrame normalizado')
# print(normalized_df)

print(data)


def menu():
    while True:
        print("\nMenú:")
        print("1. Calcular los k-vecinos de un registro seleccionado")
        print("2. Predecir la clase de un nuevo registro")
        print("3. Salir")
        choice = input("Opcion: ")

        if choice == '1':
            # Permitir al usuario seleccionar un registro de referencia
            print("\n======Comparar registro seleccionado======")
            
            while True:    
                #reference_index = int(input("Índice del registro de referencia: "))
                try:
                    reference_index = int(input("Índice del registro de referencia: "))
                    if reference_index < 0 or reference_index >= len(data):
                        print("Índice fuera de rango. Por favor, seleccione un índice válido.")
                    else:
                        break
                except ValueError:
                    print("Por favor, ingrese un número entero.")

            # Permitir al usuario especificar el valor de k
            #k = int(input("Ingrese el valor de k para calcular los k-vecinos más cercanos: "))
            while True:
                try:
                    k = int(input("Ingrese el valor de k para clasificar el nuevo registro: "))
                    if k <= 0:
                        print("Por favor, ingrese un valor de k mayor que cero.")
                    else:
                        break
                except ValueError:
                    print("Por favor, ingrese un número entero mayor que cero.")

            # Calcular k-vecinos más cercanos
            neighbors = k_nearest_neighbors(data, k, reference_index)

            # Mostrar los resultados
            print(f"\nEl registro del registro {reference_index}:")
            print(data.iloc[reference_index].to_list())

            # Mostrar los resultados
            print(f"\nLos {k} vecinos más cercanos al registro {reference_index} son:")
            for neighbor_index, distance, neighbor_class in neighbors:
                print(f"Índice: {neighbor_index}, Distancia: {distance}, Clase: {neighbor_class}")

        elif choice == '2':
            # Permitir al usuario ingresar los valores del nuevo registro
            print("\n====Ingresa los valores del nuevo registro====")
            new_point_values = []
            for column in data.columns[:-1]:  # Exclude the last column (class)
                value = float(input(f"Ingrese el valor de '{column}': "))
                new_point_values.append(value)

            # Convertir los valores del nuevo registro a un array numpy
            new_point = np.array(new_point_values)

            # Permitir al usuario especificar el valor de k
            #k = int(input("Ingrese el valor de k para clasificar el nuevo registro: "))
            while True:
                try:
                    k = int(input("Ingrese el valor de k para clasificar el nuevo registro: "))
                    if k <= 0:
                        print("Por favor, ingrese un valor de k mayor que cero.")
                    else:
                        break
                except ValueError:
                    print("Por favor, ingrese un número entero mayor que cero.")

            # Clasificar el nuevo registro
            predicted_class, k_nearest = neighbors_predict_class(data, k, new_point)

            # Mostrar el resultado
            print(f"\nLa clase predicha para el nuevo registro es: {predicted_class}")
            print(f"\nLos {k} vecinos más cercanos son:")
            for neighbor_index, distance, neighbor_class in k_nearest:
                print(f"Índice: {neighbor_index}, Distancia: {distance}, Clase: {neighbor_class}")
        
        elif choice == '3':
            print("Saliendo del programa...")
            break
    else:
        print("Opción inválida. Por favor, seleccione 1, 2 o 3.")
    
menu()

#============================Verficacion de clase========================
# # Permitir al usuario seleccionar un registro de referencia
# print("\n======Comparar registro seleccionado======")
# reference_index = int(input("Índice del registro de referencia: "))

# # Permitir al usuario especificar el valor de k
# k = int(input("Ingrese el valor de k para calcular los k-vecinos más cercanos: "))

# # Calcular k-vecinos más cercanos
# neighbors = k_nearest_neighbors(data, k, reference_index)

# # Mostrar los resultados
# print(f"\nEl registro del registro {reference_index}:")
# print(data.iloc[reference_index].to_list())

# # Mostrar los resultados
# print(f"\nLos {k} vecinos más cercanos al registro {reference_index} son:")
# for neighbor_index, distance, neighbor_class in neighbors:
#     print(f"Índice: {neighbor_index}, Distancia: {distance}, Clase: {neighbor_class}")
    
    
#==========================Predecir Clase==================================
# Permitir al usuario ingresar los valores del nuevo registro
# print("\n====Ingresa los valores del nuevo registro====")
# new_point_values = []
# for column in data.columns[:-1]:  # Exclude the last column (class)
#     value = float(input(f"Ingrese el valor de '{column}': "))
#     new_point_values.append(value)

# # Convertir los valores del nuevo registro a un array numpy
# new_point = np.array(new_point_values)

# # Permitir al usuario especificar el valor de k
# k = int(input("Ingrese el valor de k para clasificar el nuevo registro: "))

# # Clasificar el nuevo registro
# predicted_class, k_nearest = neighbors_predict_class(data, k, new_point)

# # Mostrar el resultado
# print(f"\nLa clase predicha para el nuevo registro es: {predicted_class}")
# print(f"\nLos {k} vecinos más cercanos son:")
# for neighbor_index, distance, neighbor_class in k_nearest:
#     print(f"Índice: {neighbor_index}, Distancia: {distance}, Clase: {neighbor_class}")