import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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
    
    # Count occurrences of each class in the k-nearest neighbors
    class_count = {}
    for _, dist, neighbor_class in k_nearest:
        class_count[neighbor_class] = class_count.get(neighbor_class, 0) + 1 / dist  # Weighted by distance

    # Predict the class of the new point based on the majority class among the k-nearest neighbors
    predicted_class = max(class_count, key=class_count.get)
    
    return k_nearest,predicted_class

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

def knn_train_test_split(data, k):
    """
    Train and test the k-NN algorithm on the given DataFrame with train/test split.
    """
    # Split the data into training and testing sets
    # indice = int(0.8 * data.shape[0])
    # train = data.iloc[:indice] # selecciona solo las filas hasta ese indice
    # test = data.iloc[indice:]
    
    # Split the data into training and testing sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # print('Entrenamiento')
    # print(train)
    
    # Store the indices of the test set
    test_indices = test.index.tolist()
    
    # Iterate through the test set and make predictions
    predictions = []
    for i, row in test.iterrows():
        new_point = row[:-1]
        true_class = row.iloc[-1]
        predicted_class, k_nearest = neighbors_predict_class(train, k, new_point)
        predictions.append((predicted_class, true_class, k_nearest, i))
    
    return predictions, test_indices

def find_best_k(data, max_k):
    """
    Find the best value of k for k-NN using cross-validation.
    """
    print(data)
    
    
    # Separar las características y la clase
    X = data.iloc[:, :-1]  # Todas las columnas excepto la última
    y = data.iloc[:, -1]  # La última columna es la clase
    
    # Normalizar los datos
    # df = data.copy()
    # X = normalize_data2(df)
    
    # # Convertir las características normalizadas de vuelta a DataFrame para mantener el índice alineado
    # X = pd.DataFrame(X, columns=df.columns[:-1])

    # Determinar el número de splits basado en el tamaño de la clase más pequeña
    # Mientras mas splits mas costo computacional
    min_class_size = y.value_counts().min()
    n_splits = min(5, min_class_size)

    # Lista para almacenar los puntajes de precisión
    accuracy_scores = []

    # Probar k de 1 a max_k
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        skf = StratifiedKFold(n_splits=n_splits)
        
        # Asegurarse de que k no sea mayor que el tamaño del conjunto de entrenamiento en cada fold
        valid_scores = []
        for train_index, test_index in skf.split(X, y):
            if k <= len(train_index):  # Check if k is valid for this split
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                knn.fit(X_train, y_train)
                score = knn.score(X_test, y_test)
                valid_scores.append(score)
            else:
                valid_scores.append(np.nan)  # Append NaN if k is not valid
        
        # Promediar los puntajes válidos
        if valid_scores:
            accuracy_scores.append(np.nanmean(valid_scores))
        else:
            accuracy_scores.append(np.nan)

    # Determinar el k con el puntaje de precisión más alto
    best_k = accuracy_scores.index(max(accuracy_scores)) + 1

    return best_k, accuracy_scores

def normalize_data1(data):
    # # Seleccionar solo las columnas numéricas para la normalización
    #numeric_columns = ['Variable1', 'Variable2', 'Variable3']
    numeric_columns = list(data.columns[:-1])
    numeric_data = data[numeric_columns]

    # Inicializar el objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizar los datos numéricos
    normalized_data = scaler.fit_transform(numeric_data)

    # Convertir los datos normalizados de vuelta a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=numeric_columns)

    # Agregar la columna 'Clase' al DataFrame normalizado
    normalized_df[data.columns[-1]] = data[data.columns[-1]]

    print('\nMostrar el DataFrame original')
    print(data)

    print('\nMostrar el DataFrame normalizado')
    print(normalized_df)
    
    return normalized_data

def normalize_data2(data):
    """
    Normalize numerical columns in the DataFrame and handle missing values.
    Categorical columns remain unchanged.
    """
    # Copy of dataframe exclude last colum
    df = data.iloc[:, :-1].copy()
    
    for col in df.select_dtypes(include=[np.number]):
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric values to NaN
        df[col].fillna(df[col].mean(), inplace=True)

    # Normalize numerical columns (Min-Max scaling)
    for col in df.select_dtypes(include=[np.number]):
        min_col = df[col].min()
        max_col = df[col].max()
        df[col] = (df[col] - min_col) / (max_col - min_col)

    df[data.columns[-1]] = data[data.columns[-1]]
    print(df)
    
    return df

def menu():

    global data
    print(f"\n{data}")
        
    while True:
        print("\n========================MENU=======================")
        print("1. Calcular los k-vecinos de un registro seleccionado")
        print("2. Predecir la clase de un nuevo registro")
        print("3. Mejor K (Entrenamiento y Testeo)")
        print("4. Normalizar data")
        print("5. Salir")
        choice = input("Opcion: ")

        try:
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
                neighbors,predicted_class = k_nearest_neighbors(data, k, reference_index)

                # Mostrar los resultados
                print(f"\nLos valores del registro {reference_index}:{data.iloc[reference_index].to_list()}")
                print(f'\nLa prediccion de su clase es: {predicted_class}')

                # Mostrar los resultados
                print(f"\nLos {k} vecinos más cercanos al registro {reference_index} son:")
                for neighbor_index, distance, neighbor_class in neighbors:
                    print(f"Índice: {neighbor_index}, Distancia: {distance}, Clase: {neighbor_class}")

            if choice == '2':
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
            
            if choice == '3':
                # Permitir al usuario especificar el valor de k
                # while True:
                #     try:
                #         k = int(input("Ingrese el valor de k para realizar la clasificación: "))
                #         if k <= 0:
                #             print("Por favor, ingrese un valor de k mayor que cero.")
                #         else:
                #             break
                #     except ValueError:
                #         print("Por favor, ingrese un número entero mayor que cero.")
                
                # # Realizar k-NN con conjunto de entrenamiento y prueba
                # predictions, test_indices = knn_train_test_split(data, k)

                # # Mostrar los resultados
                # print(f"\n=== Resultados del k-NN con conjunto de entrenamiento y prueba (K={k}) ===\n")
                # for i, (predicted_class, true_class, k_nearest, test_index) in enumerate(predictions):
                #     print(f"Registro de prueba {i + 1} (Índice: {test_index} : {data.iloc[test_index].tolist()})")
                #     print(f"Clase verdadera: {true_class}, Clase predicha: {predicted_class}")
                #     print(f"Vecinos más cercanos:")
                #     for neighbor_index, distance, neighbor_class in k_nearest:
                #         print(f"Índice: {neighbor_index}, Distancia: {distance:.4f}, Clase: {neighbor_class}")
                #     print("\n")
                # Encontrar el mejor valor de k
                max_k = 24
                best_k, accuracy_scores = find_best_k(data, max_k)

                print(f"\nEl mejor valor de k es: {best_k}")
                print("\nPuntajes de precisión para cada k:")
                for k in range(1, max_k + 1):
                    print(f"k={k}: {accuracy_scores[k-1]:.4f}")
                
            if choice == '4':
                data = normalize_data2(data)
                
            if choice == '5':
                print("Saliendo del programa...")
                break
        except Exception as e:
            print(e)
    
# Load data from CSV into a DataFrame
csv_file = 'temperaturaCuenca.csv'  # Reemplaza 'datos.csv' con el nombre de tu archivo CSV
data = pd.read_csv(csv_file)

if __name__ == '__main__':
    menu()
