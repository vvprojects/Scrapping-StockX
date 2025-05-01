import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import math
import pickle

def prepare_data(df):
    # Define categorical and numerical columns
    categorical_cols = ['brand', 'product_category', 'model']
    numerical_cols = [
        'annual_volatility', 'annual_sales_count', 'annual_price_premium',
        'last72h_sales_count', 'express_expedited_count', 'express_standard_count',
        'express_standard_lowest_amount'
    ]
    target_col = 'annual_avg_price'
    
    # Fill NaN values
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Prepare features and target
    X = df[categorical_cols + numerical_cols]
    y = df[target_col]
    
    return X, y, categorical_cols, numerical_cols

def train_catboost(X_train, X_val, y_train, y_val, categorical_cols):
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    
    # Initialize model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100
    )
    
    # Train model
    model.fit(
        train_pool,
        eval_set=val_pool,
        plot=True
    )
    
    return model

def evaluate_model(model, X_test, y_test, categorical_cols):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Test MAE: ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Predicted vs Actual Sneaker Prices (CatBoost)')
    plt.savefig('catboost_prediction_results.png')
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('CatBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png')
    
    return mae, rmse, r2, predictions

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('third_version_of_stockx_data_with_kinda_many_brands_without_dropping_nan_description.csv')
    
    # Prepare data
    X, y, categorical_cols, numerical_cols = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("Test data shape:", X_test.shape)
    
    # Train model
    print("\nTraining CatBoost model...")
    model = train_catboost(X_train, X_val, y_train, y_val, categorical_cols)
    
    # Evaluate model
    print("\nEvaluating model...")
    mae, rmse, r2, predictions = evaluate_model(model, X_test, y_test, categorical_cols)
    
    # Save model and metrics
    print("\nSaving model and metrics...")
    model.save_model('catboost_model.cbm')
    
    with open('catboost_metrics.txt', 'w') as f:
        f.write(f"Test MAE: ${mae:.2f}\n")
        f.write(f"Test RMSE: ${rmse:.2f}\n")
        f.write(f"Test R²: {r2:.4f}\n")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('catboost_feature_importance.csv', index=False)
    
    print("\nDone! Check the following files:")
    print("- catboost_model.cbm (trained model)")
    print("- catboost_metrics.txt (performance metrics)")
    print("- catboost_prediction_results.png (predictions plot)")
    print("- catboost_feature_importance.png (feature importance plot)")
    print("- catboost_feature_importance.csv (detailed feature importance)")

if __name__ == "__main__":
    main() 