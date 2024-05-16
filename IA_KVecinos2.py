import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        predictions = []
        distances_list = []  # Lista para almacenar las distancias calculadas
        for i in range(len(X_test)):
            distances = [self.euclidean_distance(X_test[i], x_train) for x_train in self.X_train]
            distances_list.append(distances)  # Agregar las distancias a la lista
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[j] for j in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        return predictions, distances_list

# Cargar datos desde un archivo CSV
data = pd.read_csv('kvecinos.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir los valores de K a probar
k_values = range(1, 21)  # Prueba valores de K de 1 a 20

# Realizar validación cruzada de k-fold para cada valor de K
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_error = float('inf')
best_k = None
for k in k_values:
    total_error = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        knn = KNearestNeighbors(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        total_error += 1 - accuracy_score(y_test, predictions)  # Error es 1 - precisión
        
        # Mostrar el dataframe de entrenamiento y de prueba
        print("DataFrame de entrenamiento:")
        print(pd.DataFrame(X_train))
        print("Etiquetas de entrenamiento:")
        print(pd.DataFrame(y_train))
        print("DataFrame de prueba:")
        print(pd.DataFrame(X_test))
        print("Etiquetas de prueba:")
        print(pd.DataFrame(y_test))
        
    avg_error = total_error / 5  # Promedio del error en las 5 iteraciones de validación cruzada
    print(f"Para k={k}, el error promedio fue {avg_error}")
    if avg_error < best_error:
        best_error = avg_error
        best_k = k


print(f"El mejor valor de K encontrado fue: {best_k} con una precisión promedio de: {best_error}")
