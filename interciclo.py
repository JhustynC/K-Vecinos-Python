import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def nornalizar_data(data):

    # Copiamos el dataframe menos la ultima columna
    df = data.iloc[:, :-1].copy()
    
    for col in df.select_dtypes(include=[np.number]):
         # Convertimos los valores no nuericos a NaN
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        df[col].fillna(df[col].mean(), inplace=True)

    # Normalizamos la columnas no numericas con el factor Min-Max scaling
    for col in df.select_dtypes(include=[np.number]):
        min_col = df[col].min()
        max_col = df[col].max()
        df[col] = (df[col] - min_col) / (max_col - min_col)

    df[data.columns[-1]] = data[data.columns[-1]]
    print(df)
    
    return df


def buscar_mejor_k(data, max_k):

    df = data
    # Dividir en los datos (X) y clase (y)
    X = df.drop(columns=['id', 'TipoEmpleado'])
    y = df['TipoEmpleado']

    # Dividir los datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encontrar el mejor valor de k usando el conjunto de prueba
    k_range = range(1, max_k+1)
    k_scores = []
    errors = [] #Error=1−Precision

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        error = 1 - score
        k_scores.append(score)
        errors.append(error)

    # Imprimir los resultados
    for k, score, error in zip(k_range, k_scores, errors):
        print(f"k={k}, accuracy={score:.4f}, error={error:.4f}")

    # Encontrar el valor óptimo de k
    best_k = k_range[np.argmax(k_scores)]
    print(f"El mejor valor de k es: {best_k} con una exactitud de {max(k_scores):.4f}")

    # Graficar los errores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, errors, marker='o', linestyle='dashed', color='b', label='Tasa de Error')
    plt.xlabel('Valor de k')
    plt.ylabel('Tasa de Error')
    plt.title('Tasa de Error vs. Valor de k')
    plt.xticks(k_range)
    plt.legend()
    plt.grid(True)
    plt.show()

# Cargar los datos
csv_file = 'DatosTest.csv' 
data = pd.read_csv(csv_file, delimiter=';')
df = pd.DataFrame(data)

# Codificar variables categóricas
label_encoder = LabelEncoder()

df['Casado'] = label_encoder.fit_transform(df['Casado'])
df['Carro'] = label_encoder.fit_transform(df['Carro'])
df['Alq/Prop'] = label_encoder.fit_transform(df['Alq/Prop'])
df['Sindic.'] = label_encoder.fit_transform(df['Sindic.'])
df['Sexo'] = label_encoder.fit_transform(df['Sexo'])

# Mostrar el DataFrame normalizado
data = df
print(data)
data = nornalizar_data(data)
max_k = 6
buscar_mejor_k(data, max_k)