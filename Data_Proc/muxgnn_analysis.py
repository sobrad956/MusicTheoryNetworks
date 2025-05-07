import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from itertools import combinations
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import community.community_louvain as community_louvain

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgePredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gnn = GNN(hidden_channels)
        
    def forward(self, x, edge_index):
        # Get node embeddings
        node_embeddings = self.gnn(x, edge_index)
        return node_embeddings

def evaluate_edge_prediction(model, data, edge_type):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x_dict, data.edge_index_dict)
        
        # Get embeddings for source and target nodes
        src_embeddings = node_embeddings['paper'][data[edge_type].edge_label_index[0]]
        dst_embeddings = node_embeddings['paper'][data[edge_type].edge_label_index[1]]
        
        # Compute predictions
        pred = (src_embeddings * dst_embeddings).sum(dim=-1).sigmoid()
        
        # Calculate AUC-ROC
        auc = roc_auc_score(data[edge_type].edge_label.cpu().numpy(), 
                           pred.cpu().numpy())
        
    return auc, node_embeddings['paper'].cpu().numpy()

def evaluate_clustering(embeddings, n_clusters=5):
    """Evaluate clustering quality using different methods."""
    results = {}
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
    
    # Spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
    spectral_labels = spectral.fit_predict(embeddings)
    spectral_silhouette = silhouette_score(embeddings, spectral_labels)
    
    results['kmeans'] = {
        'labels': kmeans_labels,
        'silhouette': kmeans_silhouette
    }
    
    results['spectral'] = {
        'labels': spectral_labels,
        'silhouette': spectral_silhouette
    }
    
    return results

def plot_results(edge_combinations_results, clustering_results, output_file):
    """Plot edge prediction and clustering results."""
    n_combinations = len(edge_combinations_results)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot edge prediction results
    combinations = list(edge_combinations_results.keys())
    aucs = [results['auc'] for results in edge_combinations_results.values()]
    
    sns.barplot(x=range(len(combinations)), y=aucs, ax=ax1)
    ax1.set_xticks(range(len(combinations)))
    ax1.set_xticklabels(['+'.join(comb) for comb in combinations], rotation=45, ha='right')
    ax1.set_title('Edge Prediction Performance (AUC-ROC)')
    ax1.set_ylabel('AUC-ROC Score')
    
    # Plot clustering results
    kmeans_scores = [results['clustering']['kmeans']['silhouette'] 
                    for results in edge_combinations_results.values()]
    spectral_scores = [results['clustering']['spectral']['silhouette'] 
                      for results in edge_combinations_results.values()]
    
    x = np.arange(len(combinations))
    width = 0.35
    
    ax2.bar(x - width/2, kmeans_scores, width, label='K-means')
    ax2.bar(x + width/2, spectral_scores, width, label='Spectral')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['+'.join(comb) for comb in combinations], rotation=45, ha='right')
    ax2.set_title('Clustering Quality (Silhouette Score)')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading multiplex network...")
    data = torch.load('multiplex_network.pt')
    
    # Get all edge types
    edge_types = [key for key in data.edge_index_dict.keys() 
                 if isinstance(key, tuple) and key[0] == 'paper' and key[2] == 'paper']
    
    # Generate all possible combinations of edge types
    all_combinations = []
    for r in range(1, len(edge_types) + 1):
        all_combinations.extend(combinations(edge_types, r))
    
    results = {}
    
    print("\nRunning analysis for all edge type combinations...")
    for edge_combination in tqdm(all_combinations):
        # Create a new HeteroData object with only selected edge types
        subset_data = HeteroData()
        subset_data['paper'].x = data['paper'].x
        
        for edge_type in edge_combination:
            subset_data[edge_type].edge_index = data[edge_type].edge_index
            subset_data[edge_type].edge_label_index = data[edge_type].edge_label_index
            subset_data[edge_type].edge_label = data[edge_type].edge_label
        
        # Initialize and train model
        model = EdgePredictor(hidden_channels=64)
        model = to_hetero(model, subset_data.metadata())
        
        # Evaluate edge prediction
        auc, embeddings = evaluate_edge_prediction(model, subset_data, edge_combination[0])
        
        # Evaluate clustering
        clustering_results = evaluate_clustering(embeddings)
        
        # Store results
        results[edge_combination] = {
            'auc': auc,
            'embeddings': embeddings,
            'clustering': clustering_results
        }
    
    # Plot and save results
    print("\nPlotting results...")
    plot_results(results, clustering_results, 'muxgnn_analysis.png')
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 80)
    for combination, result in results.items():
        print(f"\nEdge Types: {' + '.join(str(et) for et in combination)}")
        print(f"Edge Prediction AUC: {result['auc']:.4f}")
        print(f"K-means Silhouette: {result['clustering']['kmeans']['silhouette']:.4f}")
        print(f"Spectral Silhouette: {result['clustering']['spectral']['silhouette']:.4f}")
    print("-" * 80)

if __name__ == "__main__":
    main() 