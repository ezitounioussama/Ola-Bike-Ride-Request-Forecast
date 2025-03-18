import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_and_explore_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and display initial exploration info.
    """
    data = pd.read_csv(file_path)
    print("Initial Data Exploration:")
    print(data.head())
    print(data.info())
    print("Missing values per column:")
    print(data.isnull().sum())
    print("\n")
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and extract datetime-related features.
    """
    data.fillna(method='ffill', inplace=True)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["weekday"] = data["datetime"].dt.weekday
    data["is_weekend"] = data["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    return data


def detect_and_remove_outliers(data: pd.DataFrame, show_plots: bool = True) -> pd.DataFrame:
    """
    Detect and remove outliers using Isolation Forest on the 'count' column.
    """
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    data['outlier'] = isolation_forest.fit_predict(data[['count']])
    
    if show_plots:
        sns.boxplot(data=data, x='count', y='outlier')
        plt.title("Visualisation des outliers avec Isolation Forest")
        plt.show()
    
    # Keep only the rows where outlier == 1
    return data[data['outlier'] == 1]


def perform_clustering(data: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Perform clustering on selected weather-related features and add the cluster labels to the data.
    """
    cluster_features = ["temp", "atemp", "humidity", "windspeed"]
    # Keep only the features that exist in the dataset
    cluster_features = [cf for cf in cluster_features if cf in data.columns]
    
    if cluster_features:
        X_cluster = data[cluster_features].copy()
        X_cluster.fillna(method='ffill', inplace=True)
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(X_cluster_scaled)
    else:
        data['cluster'] = 0
    return data


def prepare_features(data: pd.DataFrame, show_plots: bool = True):
    """
    Convert categorical features to dummy variables and split the data into features and target.
    Optionally, display the correlation matrix.
    """
    if show_plots:
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice de Corrélation")
        plt.show()
    
    # Convert categorical variables (e.g., season, weather, cluster) into dummy variables
    data = pd.get_dummies(data, columns=["season", "weather"], drop_first=True)
    data = pd.get_dummies(data, columns=["cluster"], prefix="clust", drop_first=True)
    
    X = data.drop(columns=["datetime", "count", "outlier"])
    y = data["count"]
    return X, y


def split_and_scale(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets and apply standard scaling.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor with the predetermined best parameters.
    Best Parameters: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
    """
    model = RandomForestRegressor(max_depth=20, min_samples_split=2, n_estimators=200, random_state=42)
    print("Using Random Forest Regressor with best parameters:")
    print(model)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, show_plots: bool = True):
    """
    Evaluate the model using MAE and RMSE, and optionally plot predictions vs. actual values.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title('Prédictions vs Valeurs Réelles (Random Forest)')
        plt.show()
    
    return mae, rmse


def process_and_predict(file_path: str, n_clusters: int = 3, show_plots: bool = True):
    """
    Process the dataset and predict ride-request counts using a Random Forest Regressor.
    
    Args:
        file_path (str): Path to the CSV file.
        n_clusters (int): Number of clusters to form during clustering.
        show_plots (bool): Whether to display plots.
    
    Returns:
        model: The trained Random Forest model.
        mae (float): Mean Absolute Error on the test set.
        rmse (float): Root Mean Squared Error on the test set.
    """
    data = load_and_explore_data(file_path)
    data = preprocess_data(data)
    
    if show_plots:
        sns.histplot(data["count"], bins=30, kde=True)
        plt.title('Distribution des demandes de vélo')
        plt.show()
    
    data = detect_and_remove_outliers(data, show_plots=show_plots)
    data = perform_clustering(data, n_clusters=n_clusters)
    X, y = prepare_features(data, show_plots=show_plots)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    model = train_model(X_train, y_train)
    mae, rmse = evaluate_model(model, X_test, y_test, show_plots=show_plots)
    
    return model, mae, rmse


if __name__ == "__main__":
    file_path = "/content/train.csv"  # Update this path as needed
    model, mae, rmse = process_and_predict(file_path, n_clusters=3, show_plots=True)
