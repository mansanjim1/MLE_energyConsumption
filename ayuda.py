import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
df['Holiday'] = df['Holiday'].map({'No': 0, 'Yes': 1})
df['DayOfWeek'] = df['DayOfWeek'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
df['HVACUsage'] = df['HVACUsage'].map({'Off': 0, 'On': 1})
df['LightingUsage'] = df['LightingUsage'].map({'Off': 0, 'On': 1})

# Normalización de datos
columns_to_exclude = ['Month', 'Hour', 'DayOfWeek', 'Holiday', 'HVACUsage', 'season', 'Occupancy']
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(columns_to_exclude)
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Selección de características
selected_features = ['SquareFootage', 'Temperature', 'RenewableEnergy', 'Humidity', 'Hour', 'Occupancy', 'HVACUsage', 'Month', 'LightingUsage', 'DayOfWeek', 'Holiday']
ordered_features = ['EnergyConsumption'] + selected_features
df_ordered = df[ordered_features]

# División del conjunto de datos
X = df_ordered.drop(columns=['EnergyConsumption'])
y = df_ordered['EnergyConsumption']
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

# Optimización de hiperparámetros para KNN
param_grid_knn = {'n_neighbors': [3, 5, 10, 15, 20], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

print("Mejores hiperparámetros para KNN:", grid_search_knn.best_params_)

# Crear modelo KNN con los mejores hiperparámetros
best_params = grid_search_knn.best_params_
model_knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], p=best_params['p'], weights=best_params['weights'])
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("KNN - MSE:", mse_knn)
print("KNN - R²:", r2_knn)

# Optimización de hiperparámetros para XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

print("Mejores hiperparámetros para XGBoost:", grid_search_xgb.best_params_)

# Modelo XGBoost con los mejores hiperparámetros
best_params = grid_search_xgb.best_params_
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost - MSE:", mse_xgb)
print("XGBoost - R²:", r2_xgb)

# Crear el modelo de red neuronal
model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Capa de salida para regresión
])

# Compilar el modelo
model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
model_nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluación del modelo
mse_nn, mae_nn = model_nn.evaluate(X_test, y_test, verbose=0)
print("Red Neuronal - MSE:", mse_nn)

# Visualización de métricas
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
