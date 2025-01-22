import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Cargar datos
df = pd.read_csv('Data/Energy_consumption_dataset.csv')

# Exploración de datos
print("Información del dataframe:")
print(df.info(), "\n")

print("Primeras instancias del dataframe:")
print(df.head(), "\n")

print("Descripción del dataframe:")
print(df.describe())

print("Se revisa si existe algún dato faltante")
print(df.isnull().sum())

# Visualización de la distribución
df['EnergyConsumption'].hist(bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribución de consumo de energía')
plt.xlabel('Consumo de energía')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Función para asignar estaciones del año
def get_season(month):
    if month in [12, 1, 2]:
        return 'Invierno'
    elif month in [3, 4, 5]:
        return 'Primavera'
    elif month in [6, 7, 8]:
        return 'Verano'
    elif month in [9, 10, 11]:
        return 'Otoño'


df['season'] = df['Month'].apply(get_season)

# Conversión de datos categóricos a numéricos
df = pd.get_dummies(df, columns=['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage'], drop_first=True)

# Seleccionar solo las columnas numéricas
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Crear boxplots para cada variable numérica para comprobar los outliers, el IQR y la mediana
plt.figure(figsize=(12, 8))
df[numeric_columns].boxplot()
plt.title('Boxplot de variables numéricas')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Normalización de datos
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(['EnergyConsumption'])
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

X = df.drop(columns=['EnergyConsumption','season']) #Quitamos season porque no es necesario para los modelos
y = df['EnergyConsumption']

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Modelos de regresión sin optimizacion
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Regresión Lineal - MSE:", mse_lr)
print("Regresión Lineal - R²:", r2_lr)

# Árbol de decisión
model_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print("Árbol de Decisión - MSE:", mse_tree)
print("Árbol de Decisión - R²:", r2_tree)

# K-Nearest Neighbors
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("K-Nearest Neighbors - MSE:", mse_knn)
print("K-Nearest Neighbors - R²:", r2_knn)

model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost - MSE:", mse_xgb)
print("XGBoost - R²:", r2_xgb)


# Crear gráfico comparativo de MSE y R²
models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']
mse_values = [mse_lr, mse_tree, mse_knn, mse_xgb]
r2_values = [r2_lr, r2_tree, r2_knn, r2_xgb]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(models, r2_values, color='blue', alpha=0.7, label='R²')
ax1.set_ylabel('R²', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0, 1)  # R² suele estar en el rango de 0 a 1
for i, r2 in enumerate(r2_values):
    ax1.text(i, r2, f'{r2:.2f}', ha='center', va='bottom', fontsize=10, color='black')
ax2 = ax1.twinx()
ax2.plot(models, mse_values, color='red', marker='o', label='MSE', linewidth=2)
ax2.set_ylabel('MSE', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title('Comparación de MSE y R² entre modelos')
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Hiperparámetros para Regresión Lineal
param_grid_lr = {
    'fit_intercept': [True, False],  # Si se ajusta el intercepto o no
    'copy_X': [True, False]          # Si se copia X o se sobrescribe
}

# Hiperparámetros para Árbol de Decisión
param_grid_tree = {
    'max_depth': [3, 5, 10, 20],          # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],      # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 5]         # Mínimo de muestras por hoja
}

# Hiperparámetros para K-Nearest Neighbors (KNN)
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],          # Número de vecinos
    'weights': ['uniform', 'distance'],     # Peso uniforme o basado en la distancia
    'metric': ['euclidean', 'manhattan']    # Distancia euclidiana o manhattan
}

# Hiperparámetros para XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],         # Número de árboles
    'learning_rate': [0.01, 0.1, 0.2],      # Tasa de aprendizaje
    'max_depth': [3, 5, 7]                  # Profundidad máxima del árbol
}
# Aplicar GridSearchCV a cada modelo
grid_search_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, scoring='neg_mean_squared_error')
grid_search_tree = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=5, scoring='neg_mean_squared_error')
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid_xgb, cv=5, scoring='neg_mean_squared_error')

# Entrenar modelos optimizados
grid_search_lr.fit(X_train, y_train)
grid_search_tree.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)

# Obtener los mejores modelos
best_lr = grid_search_lr.best_estimator_
best_tree = grid_search_tree.best_estimator_
best_knn = grid_search_knn.best_estimator_
best_xgb = grid_search_xgb.best_estimator_

# Realizar predicciones en base a los mejores modelos obtenidos anteriormente
y_pred_lr_gs = best_lr.predict(X_test)
y_pred_tree_gs = best_tree.predict(X_test)
y_pred_knn_gs = best_knn.predict(X_test)
y_pred_xgb_gs = best_xgb.predict(X_test)

# Calcular R² de los modelos optimizados
r2_lr_gs = r2_score(y_test, y_pred_lr_gs)
r2_tree_gs = r2_score(y_test, y_pred_tree_gs)
r2_knn_gs = r2_score(y_test, y_pred_knn_gs)
r2_xgb_gs = r2_score(y_test, y_pred_xgb_gs)

mse_values = [
    mean_squared_error(y_test, y_pred_lr_gs),
    mean_squared_error(y_test, y_pred_tree_gs),
    mean_squared_error(y_test, y_pred_knn_gs),
    mean_squared_error(y_test, y_pred_xgb_gs)
]

print("Regresión Lineal con Hiper-parametros ajustados - MSE:", mse_values[0])
print("Regresión Lineal con Hiper-parametros ajustados - R²:", r2_lr_gs)
print("Árbol de Decisión con Hiper-parametros ajustados- MSE:", mse_values[1])
print("Árbol de Decisión con Hiper-parametros ajustados- R²:", r2_tree_gs)
print("K-Nearest Neighbors con Hiper-parametros ajustados - MSE:", mse_values[2])
print("K-Nearest Neighbors con Hiper-parametros ajustados - R²:", r2_knn_gs)
print("XGBoost con Hiper-parametros ajustados - MSE:", mse_values[3])
print("XGBoost con Hiper-parametros ajustados - R²:", r2_xgb_gs)

r2_values = [
    r2_score(y_test, y_pred_lr_gs),
    r2_score(y_test, y_pred_tree_gs),
    r2_score(y_test, y_pred_knn_gs),
    r2_score(y_test, y_pred_xgb_gs)
]

models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']

# Visualización de los resultados despues de optimizar los hiper-parametros
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(models, r2_values, color='blue', alpha=0.7, label='R²')
ax1.set_ylabel('R²', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0, 1)
for i, r2 in enumerate(r2_values):
    ax1.text(i, r2, f'{r2:.2f}', ha='center', va='bottom', fontsize=10, color='black')

ax2 = ax1.twinx()
ax2.plot(models, mse_values, color='red', marker='o', label='MSE', linewidth=2)
ax2.set_ylabel('MSE', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title('Comparación de MSE y R² entre modelos optimizados')
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Comparación de R² antes y después de la optimización
models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']
r2_before = [r2_lr, r2_tree, r2_knn, r2_xgb]
r2_after = [r2_lr_gs, r2_tree_gs, r2_knn_gs, r2_xgb_gs]

# Visualización de los resultados
x = range(len(models))
plt.figure(figsize=(10, 6))
plt.bar(x, r2_before, width=0.4, label='Sin GridSearch', color='blue', align='center')
plt.bar([p + 0.4 for p in x], r2_after, width=0.4, label='Con GridSearch', color='green', align='center')
plt.xticks([p + 0.2 for p in x], models)
plt.ylabel('R²')
plt.title('Comparación de R² antes y después de GridSearch')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Seleccionar las mejores características
# Seleccionar las 10 mejores características usando SelectKBest
k = 10  # Número de características a seleccionar
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)

# Nombres de las características seleccionadas
selected_features = X.columns[selector.get_support()]
print(f'Mejores {k} características seleccionadas:', selected_features)

# Crear nuevo DataFrame con las características seleccionadas
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# División del conjunto de datos con las características seleccionadas
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected_df, y, test_size=0.20, random_state=42)

# Aplicar GridSearchCV a cada modelo con las características seleccionadas
grid_search_lr_sel = GridSearchCV(LinearRegression(), param_grid_lr, cv=5, scoring='r2')
grid_search_tree_sel = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=5, scoring='r2')
grid_search_knn_sel = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, scoring='r2')
grid_search_xgb_sel = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid_xgb, cv=5, scoring='r2')

# Entrenar modelos con GridSearch
grid_search_lr_sel.fit(X_train_sel, y_train_sel)
grid_search_tree_sel.fit(X_train_sel, y_train_sel)
grid_search_knn_sel.fit(X_train_sel, y_train_sel)
grid_search_xgb_sel.fit(X_train_sel, y_train_sel)

# Obtener los mejores modelos con características seleccionadas
best_lr_sel = grid_search_lr_sel.best_estimator_
best_tree_sel = grid_search_tree_sel.best_estimator_
best_knn_sel = grid_search_knn_sel.best_estimator_
best_xgb_sel = grid_search_xgb_sel.best_estimator_

# Realizar predicciones con los modelos optimizados
y_pred_lr_sel = best_lr_sel.predict(X_test_sel)
y_pred_tree_sel = best_tree_sel.predict(X_test_sel)
y_pred_knn_sel = best_knn_sel.predict(X_test_sel)
y_pred_xgb_sel = best_xgb_sel.predict(X_test_sel)

# Calcular R² de los modelos optimizados con características seleccionadas
r2_lr_sel = r2_score(y_test_sel, y_pred_lr_sel)
r2_tree_sel = r2_score(y_test_sel, y_pred_tree_sel)
r2_knn_sel = r2_score(y_test_sel, y_pred_knn_sel)
r2_xgb_sel = r2_score(y_test_sel, y_pred_xgb_sel)

# Comparación de R² con características seleccionadas
models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']
r2_values_selected = [r2_lr_sel, r2_tree_sel, r2_knn_sel, r2_xgb_sel]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_values_selected, color='blue', alpha=0.7, label='R² con KBest')
plt.ylabel('R²')
plt.title('Comparación de R² con mejores características seleccionadas')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar valores sobre las barras
for i, r2 in enumerate(r2_values_selected):
    plt.text(i, r2, f'{r2:.2f}', ha='center', va='bottom', fontsize=10, color='black')

plt.show()

# A continuación empieza la parte de clustering

# Aplicar PCA para reducir a n componentes principales

# Evaluar la inercia para diferentes valores de k, gracias a esto podemos obtener el número de clusteres optimo
inertia = []
range_clusters = range(2, 11)  # Probar de 2 a 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range_clusters, inertia, marker='o', linestyle='-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.grid()
plt.show()

# Iterar sobre diferentes números de componentes principales
for i in range(2, 5):
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X)

    # Aplicar K-Means después de la reducción de dimensionalidad
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    # Visualizar los clusters resultantes (usando solo las dos primeras componentes para visualización)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50)
    plt.title(f'Clustering con K-Means después de PCA ({i} Componentes)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.legend(title='Clusters')
    plt.show()

    # Calcular y mostrar el coeficiente de Silhouette
    sil_score = silhouette_score(X_pca, clusters)
    print(f'Número de Componentes: {i} - Coeficiente de Silhouette: {sil_score:.2f}')
