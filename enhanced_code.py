import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)

import pandas as pd
pd.set_option('display.max_colwidth', None)  # Show full content in DataFrame columns

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm

###############################################################################
# 1) HELPER FUNCTIONS FOR DATA PREPARATION
###############################################################################

def prepare_train_data(df, n_clusters=3, show_plots=False):
    """
    Preprocess training data:
      - Drop target-leaking columns if present (casual, registered).
      - Convert datetime, create time-based features.
      - Detect outliers (IsolationForest) based on 'count'.
      - Drop outliers.
      - Perform KMeans clustering on selected weather features.
      - One-hot encode season/weather/cluster.
      - Return preprocessed DataFrame ready for splitting into X, y.
      - Also return the fitted StandardScaler, fitted KMeans, and the final columns used for X.
    """
    # 1) Drop leaking columns if they exist
    for col in ["casual", "registered"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # 2) Basic cleaning
    df.fillna(method='ffill', inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    
    # Optional: show distribution
    if show_plots:
        sns.histplot(df["count"], bins=30, kde=True)
        plt.title("Distribution of 'count'")
        plt.show()
    
    # 3) Outlier detection
    if "count" in df.columns:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["outlier"] = iso.fit_predict(df[["count"]])
        df = df[df["outlier"] == 1]
        df.drop(columns=["outlier"], inplace=True)
    
    # 4) Clustering
    cluster_features = ["temp", "atemp", "humidity", "windspeed"]
    cluster_features = [c for c in cluster_features if c in df.columns]
    if len(cluster_features) > 0:
        # Fit KMeans on these features
        X_cluster = df[cluster_features].copy()
        X_cluster.fillna(method='ffill', inplace=True)
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(X_cluster_scaled)
    else:
        kmeans = None
        df["cluster"] = 0
    
    # 5) One-hot encoding
    for col_to_dummy in ["season", "weather", "cluster"]:
        if col_to_dummy in df.columns:
            df = pd.get_dummies(df, columns=[col_to_dummy], prefix=col_to_dummy, drop_first=True)
    
    # 6) Final feature selection
    if "count" not in df.columns:
        raise ValueError("No 'count' column found in the training data!")
    y = df["count"]
    
    # Drop columns not used as features
    # We keep everything except 'datetime' and 'count'
    drop_cols = ["datetime", "count"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 7) Scale the final X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return the processed data plus fitted objects needed for test set
    return X_scaled, y, scaler, kmeans, X.columns.tolist()


def prepare_test_data(df, train_columns, scaler, kmeans=None):
    """
    Preprocess test data similarly to train:
      - Convert datetime, create time-based features.
      - NO outlier removal (no 'count').
      - Use the same KMeans model from train to predict cluster labels.
      - One-hot encode season/weather/cluster, then align columns with train.
      - Scale using the same scaler from train.
    """
    # 1) Basic cleaning
    df.fillna(method='ffill', inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2) Cluster prediction if possible
    cluster_features = ["temp", "atemp", "humidity", "windspeed"]
    cluster_features = [c for c in cluster_features if c in df.columns]
    if kmeans is not None and len(cluster_features) > 0:
        # Use the same cluster approach from train
        X_cluster = df[cluster_features].copy()
        X_cluster.fillna(method='ffill', inplace=True)
        # We assume KMeans was fitted on scaled data, so we must scale here the same way:
        local_scaler = StandardScaler()
        X_cluster_scaled = local_scaler.fit_transform(X_cluster)
        df["cluster"] = kmeans.predict(X_cluster_scaled)
    else:
        df["cluster"] = 0
    
    # 3) One-hot encoding
    for col_to_dummy in ["season", "weather", "cluster"]:
        if col_to_dummy in df.columns:
            df = pd.get_dummies(df, columns=[col_to_dummy], prefix=col_to_dummy, drop_first=True)
    
    # 4) Align columns with train
    # We do NOT have 'count' in test
    # Also 'datetime' is needed for submission, so we won't drop it from the DataFrame itself.
    test_cols = df.columns.tolist()
    # We only want the columns that were used for training
    # but some might not exist in test, so we align them:
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0  # add missing column
    # Also remove any extra columns that train didn't have (like 'count' if it existed)
    extra_cols = set(df.columns) - set(train_columns) - {"datetime"}
    if extra_cols:
        df.drop(columns=extra_cols, inplace=True)
    
    # Reorder columns to match the exact order of train_columns
    df = df.reindex(columns=["datetime"] + train_columns)  # keep datetime in front
    
    # 5) Scale using the training scaler
    X_test = df[train_columns].values
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_test_scaled


###############################################################################
# 2) MAIN SCRIPT
###############################################################################
def main():
    # Filenames (adjust paths if needed)
    train_file = "train.csv"
    test_file = "test.csv"
    submission_file = "sampleSubmission.csv"
    
    # -----------------------------
    # A) Load and preprocess TRAIN
    # -----------------------------
    train_df = pd.read_csv(train_file)
    
    # Prepare train data (includes outlier removal, clustering, etc.)
    X_scaled, y, scaler, kmeans, train_columns = prepare_train_data(train_df, n_clusters=3, show_plots=False)
    
    # -----------------------------
    # B) Local train/test split for evaluation
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # We’ll try multiple models
    model_types = ['random_forest', 'xgboost', 'lightgbm']
    best_model = None
    best_model_name = None
    best_mae = float('inf')
    best_model_params = None
    results = []
    
    for mtype in model_types:
        print("=====================================")
        print(f"Training model: {mtype}")
        
        # Create model
        if mtype == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        elif mtype == 'xgboost':
            model = xgb.XGBRegressor(random_state=42)
        elif mtype == 'lightgbm':
            model = lgb.LGBMRegressor(verbosity=-1, random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
        }
        print("Tuning hyperparameters...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params_ = grid_search.best_params_
        best_model_ = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_pred_val = best_model_.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_pred_val)
        
        # Adjusted R² (statsmodels approach)
        X_pred_const = sm.add_constant(y_pred_val)
        ols_model = sm.OLS(y_val, X_pred_const).fit()
        adj_r2_val = ols_model.rsquared_adj
        
        print(f"MAE = {mae_val:.4f}, RMSE = {rmse_val:.4f}, R² = {r2_val:.4f}, Adjusted R² = {adj_r2_val:.4f}")
        print(f"Best parameters: {best_params_}")
        
        # Keep track
        results.append({
            "Model": mtype,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "R²": r2_val,
            "Adjusted R²": adj_r2_val,
            "Best Parameters": best_params_
        })
        
        # Update best model
        if mae_val < best_mae:
            best_mae = mae_val
            best_model = best_model_
            best_model_name = mtype
            best_model_params = best_params_
        print()
    
    # Summarize results
    results_df = pd.DataFrame(results)
    print("Résumé des résultats:")
    print(results_df)
    print("\n")
    print(f"Le meilleur modèle est {best_model_name} avec MAE: {best_mae} et Best Parameters: {best_model_params}\n")
    
    # -----------------------------
    # C) Retrain on the ENTIRE train set with the best model
    # -----------------------------
    best_model.fit(X_scaled, y)  # re-fit using all training data
    
    # -----------------------------
    # D) Preprocess TEST data and predict
    # -----------------------------
    test_df = pd.read_csv(test_file)
    # We transform the test data with the same pipeline (except outlier removal)
    test_df_processed, X_test_scaled = prepare_test_data(test_df, train_columns, scaler, kmeans=kmeans)
    
    # Predict on test
    y_test_pred = best_model.predict(X_test_scaled)
    
    # -----------------------------
    # E) Create submission file
    # -----------------------------
    # We'll read the sampleSubmission and fill in the "count" column
    submission_df = pd.read_csv(submission_file)
    # Match up the rows by the same datetime if needed, or if they're in the same order
    # Typically, Kaggle expects the same order as test.csv
    submission_df["count"] = y_test_pred
    
    # Save to new CSV
    submission_df.to_csv("my_submission.csv", index=False)
    print("Final submission saved to 'my_submission.csv'.")


if __name__ == "__main__":
    main()
