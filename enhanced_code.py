import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

def process_and_predict(file_path, model_type='random_forest', tune=False):
    # Charger les données
    data = pd.read_csv(file_path)
    
    # Exploration initiale
    print("Initial Data Exploration:")
    print(data.head())
    print(data.info())
    print(data.isnull().sum())

    # Gestion des valeurs manquantes
    data.fillna(method='ffill', inplace=True)  # Utilisation de la méthode forward fill
    
    # Nettoyage et transformation des données
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["is_weekend"] = data["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    
    # Visualiser la distribution des valeurs
    sns.histplot(data["count"], bins=30, kde=True)
    plt.title('Distribution des demandes de vélo')
    plt.show()

    # Détection des outliers avec Isolation Forest (Avancé)
    isolation_forest = IsolationForest(contamination=0.05)  # 5% outliers
    data['outlier'] = isolation_forest.fit_predict(data[['count']])
    
    # Visualiser les outliers
    sns.boxplot(data=data, x='count', y='outlier')
    plt.title("Visualisation des outliers avec Isolation Forest")
    plt.show()

    # Supprimer les outliers (valeurs qui sont détectées comme -1)
    data = data[data['outlier'] == 1]
    
    # Corrélation entre les variables numériques
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de Corrélation")
    plt.show()

    # Préparation des données pour l'entraînement
    data = pd.get_dummies(data, columns=["season", "weather"], drop_first=True)
    
    # Sélectionner les features et target
    X = data.drop(columns=["datetime", "count", "outlier"])  # Exclure la cible et les features inutiles
    y = data["count"]

    # Séparer les données en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choisir le modèle à utiliser
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        print("Using Random Forest Regressor.")
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
        print("Using XGBoost Regressor.")
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(random_state=42)
        print("Using LightGBM Regressor.")
    else:
        print("Invalid model type! Using Random Forest as default.")
        model = RandomForestRegressor(random_state=42)

    # Hyperparameter Tuning with GridSearchCV (if tune=True)
    if tune:
        print("Tuning hyperparameters...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("Best parameters found:", grid_search.best_params_)
        model = grid_search.best_estimator_

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    
    # Calculer les erreurs
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    # Visualiser les résultats
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
    plt.xlabel('Vrais valeurs')
    plt.ylabel('Prédictions')
    plt.title('Prédictions vs Réel')
    plt.show()
    
    return model, mae, rmse

# Utilisation de la fonction
file_path = "./train.csv"
model, mae, rmse = process_and_predict(file_path, model_type='xgboost', tune=True)
