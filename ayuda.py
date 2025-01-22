import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE

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

# # Visualización de la distribución
# df['EnergyConsumption'].hist(bins=30, color='blue', alpha=0.7, edgecolor='black')
# plt.title('Distribución de consumo de energía')
# plt.xlabel('Consumo de energía')
# plt.ylabel('Frecuencia')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()


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

# Feature engineering
df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']
df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df = pd.get_dummies(df, columns=['TimeOfDay'], drop_first=True)

# Eliminación de valores atípicos
Q1 = df['EnergyConsumption'].quantile(0.25)
Q3 = df['EnergyConsumption'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['EnergyConsumption'] >= (Q1 - 1.5 * IQR)) & (df['EnergyConsumption'] <= (Q3 + 1.5 * IQR))]

# Normalización de datos
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(['EnergyConsumption'])
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Selección de características
selected_features = ['SquareFootage', 'Temperature', 'RenewableEnergy', 'Humidity', 'Hour', 'Occupancy', 'HVACUsage_On',
                     'Month', 'LightingUsage_On', 'DayOfWeek_Tuesday', 'Holiday_Yes', 'Temp_Humidity_Interaction',
                     'TimeOfDay_Morning', 'TimeOfDay_Afternoon', 'TimeOfDay_Evening']
ordered_features = ['EnergyConsumption'] + selected_features
df_ordered = df[ordered_features]

# Balanceo del dataset con SMOTE
X = df_ordered.drop(columns=['EnergyConsumption'])
y = df_ordered['EnergyConsumption']

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Modelos de regresión
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

# Optimización de hiperparámetros para XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid_xgb, cv=5,
                               scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

# # Modelo XGBoost con los mejores hiperparámetros
best_params = grid_search_xgb.best_params_
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost - MSE:", mse_xgb)
print("XGBoost - R²:", r2_xgb)

models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']
mse_values = [mse_lr, mse_tree, mse_knn, mse_xgb]
r2_values = [r2_lr, r2_tree, r2_knn, r2_xgb]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, alpha=0.7, label='R²')
plt.plot(models, mse_values, color='red', marker='o', label='MSE')
plt.title('Comparación de MSE y R² entre modelos')
plt.ylabel('Valor')
plt.legend()
plt.show()

X = df_ordered.drop(columns=['EnergyConsumption'])
y = df_ordered['EnergyConsumption']

# Seleccionar las 10 características más importantes según ANOVA F-value
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X, y)

# Mostrar las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("Características seleccionadas:", selected_features)

X_selected_df = X[selected_features]

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.20, random_state=0)

# Modelos de regresión
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Regresión Lineal - MSE con características seleccionadas:", mse_lr)
print("Regresión Lineal - R² con características seleccionadas:", r2_lr)

from sklearn.cluster import DBSCAN, KMeans

X_clustering = df_ordered.drop(columns=['EnergyConsumption'])

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
df_ordered['Cluster_DBSCAN'] = dbscan.fit_predict(X_clustering)

# Contar el número de clusters encontrados (valor -1 indica ruido/outliers)
print(df_ordered['Cluster_DBSCAN'].value_counts())

# Visualizar los clusters
sns.scatterplot(x=df_ordered['SquareFootage'], y=df_ordered['EnergyConsumption'], hue=df_ordered['Cluster_DBSCAN'],
                palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# Seleccionar las características para el clustering (sin la variable objetivo)
X_clustering = df_ordered.drop(columns=['EnergyConsumption'])

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
range_clusters = range(2, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clustering)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia (Distorsión)')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.grid()
plt.show()

# Selección de características usando SelectKBest
X = df_ordered.drop(columns=['EnergyConsumption'])
y = df_ordered['EnergyConsumption']
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Características seleccionadas con KBest:", selected_features)

X_selected_df = X[selected_features]

# Clustering con K-Means usando las 5 características más importantes
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df_ordered['Cluster_KMeans'] = kmeans.fit_predict(X_selected_df)

# Visualización de los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_ordered[selected_features[0]],
    y=df_ordered[selected_features[1]],
    hue=df_ordered['Cluster_KMeans'],
    palette='viridis',
    alpha=0.7,
    s=50
)
plt.title(f'Clustering K-Means con {optimal_clusters} Clusters')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.legend(title="Clusters")
plt.grid(True)
plt.show()

# Evaluación del clustering
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X_selected_df, df_ordered['Cluster_KMeans'])
print(f'Coeficiente de Silhouette para {optimal_clusters} clusters: {silhouette_avg:.2f}')

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Aplicar PCA para reducir la dimensionalidad a 2 componentes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected_df)

# Convertir el resultado en un DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Aplicar K-Means a los datos reducidos
optimal_clusters = 4  # O prueba diferentes valores
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df_pca['Cluster'] = kmeans.fit_predict(X_pca)

# Visualización de los clusters resultantes
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df_pca['Cluster'], palette='viridis', s=50)
plt.title('Clustering K-Means con PCA (2 Componentes Principales)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clusters')
plt.grid(True)
plt.show()

pca_full = PCA()
pca_full.fit(X_selected_df)

# Visualizar la varianza explicada acumulativa
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada Explicada')
plt.title('Selección del número óptimo de componentes PCA')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X_pca, df_pca['Cluster'])
print(f'Coeficiente de Silhouette después de PCA: {silhouette_avg:.2f}')

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.20, random_state=42)
# Crear y entrenar el modelo de regresión lineal
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

# Optimización de hiperparámetros para XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid_xgb, cv=5,
                               scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

# Modelo XGBoost con los mejores hiperparámetros
best_params = grid_search_xgb.best_params_
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost - MSE:", mse_xgb)
print("XGBoost - R²:", r2_xgb)

models = ['Regresión Lineal', 'Árbol de Decisión', 'KNN', 'XGBoost']
mse_values = [mse_lr, mse_tree, mse_knn, mse_xgb]
r2_values = [r2_lr, r2_tree, r2_knn, r2_xgb]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, alpha=0.7, label='R²')
plt.plot(models, mse_values, color='red', marker='o', label='MSE')
plt.title('Comparación de MSE y R² entre modelos')
plt.ylabel('Valor')
plt.legend()
plt.show()

X = df_ordered.drop(columns=['EnergyConsumption'])
y = df_ordered['EnergyConsumption']

# Seleccionar las 10 características más importantes según ANOVA F-value
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X, y)

# Mostrar las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("Características seleccionadas:", selected_features)

X_selected_df = X[selected_features]

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.20, random_state=0)

# Modelos de regresión
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Regresión Lineal - MSE con características seleccionadas:", mse_lr)
print("Regresión Lineal - R² con características seleccionadas:", r2_lr)


#import pandas as pd
#from mlxtend.frequent_patterns import apriori, association_rules
#
## Convertir las variables categóricas en variables binarias
#df_association = df_ordered.copy()
#
## Convertir variables numéricas en categorías binarias (ejemplo: si Temperature > 24 se marca como 1)
#df_association['High_Temperature'] = (df_association['Temperature'] > 24).astype(int)
#df_association['High_Humidity'] = (df_association['Humidity'] > 50).astype(int)
#df_association['High_Occupancy'] = (df_association['Occupancy'] > 5).astype(int)
#
## Eliminar columnas innecesarias
#df_association.drop(columns=['EnergyConsumption', 'Temperature', 'Humidity', 'Occupancy'], inplace=True)
#
## Mostrar la estructura del nuevo dataframe binarizado
#print(df_association.head())
#