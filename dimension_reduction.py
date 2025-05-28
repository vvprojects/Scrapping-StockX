import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_raw_features():
    """Load raw InceptionV3 features."""
    with open('raw_inception_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def reduce_dimensions(features_dict, n_components_pca=10, n_components_umap=8):
    """Reduce dimensions using PCA and UMAP."""
    # Convert dictionary to array
    features_array = np.array(list(features_dict.values()))
    url_keys = list(features_dict.keys())
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Apply PCA
    pca = PCA(n_components=n_components_pca)
    pca_features = pca.fit_transform(features_scaled)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=n_components_umap, random_state=42)
    umap_features = reducer.fit_transform(features_scaled)
    
    # Create DataFrames
    pca_df = pd.DataFrame(pca_features, index=url_keys)
    pca_df.columns = [f'pca_{i+1}' for i in range(n_components_pca)]
    pca_df.index.name = 'url_key'
    
    umap_df = pd.DataFrame(umap_features, index=url_keys)
    umap_df.columns = [f'umap_{i+1}' for i in range(n_components_umap)]
    umap_df.index.name = 'url_key'
    
    # Analyze PCA components
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA Analysis:")
    print(f"Total explained variance: {cumulative_variance[-1]:.2%}")
    print("\nTop 5 PCA components explained variance:")
    for i, var in enumerate(explained_variance[:5]):
        print(f"Component {i+1}: {var:.2%}")
    
    # Visualize PCA components
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.savefig('pca_variance.png')
    plt.close()
    
    # Visualize UMAP components
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_features[:, 0], umap_features[:, 1], alpha=0.5)
    plt.title('UMAP Visualization of Image Features')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig('umap_visualization.png')
    plt.close()
    
    return pca_df, umap_df, pca, reducer

def main():
    # Load raw features
    print("Loading raw features...")
    features_dict = load_raw_features()
    
    # Create output directory for experiments
    os.makedirs('dimension_reduction_experiments', exist_ok=True)
    
    # Experiment with different numbers of components
    pca_components = [10]
    umap_components = [8]
    
    for n_pca in pca_components:
        for n_umap in umap_components:
            print(f"\nExperimenting with PCA={n_pca}, UMAP={n_umap}")
            
            # Reduce dimensions
            pca_df, umap_df, pca, reducer = reduce_dimensions(
                features_dict,
                n_components_pca=n_pca,
                n_components_umap=n_umap
            )
            
            # Save features
            experiment_name = f"pca{n_pca}_umap{n_umap}"
            pca_df.to_csv(f'dimension_reduction_experiments/image_features_pca_{experiment_name}.csv')
            umap_df.to_csv(f'dimension_reduction_experiments/image_features_umap_{experiment_name}.csv')
            
            # Save models
            with open(f'dimension_reduction_experiments/pca_model_{experiment_name}.pkl', 'wb') as f:
                pickle.dump(pca, f)
            with open(f'dimension_reduction_experiments/umap_model_{experiment_name}.pkl', 'wb') as f:
                pickle.dump(reducer, f)
            
            print(f"Saved experiment results for {experiment_name}")

if __name__ == "__main__":
    main() 
