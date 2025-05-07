import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import gzip
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import community.community_louvain as community_louvain
import matplotlib.ticker as mtick
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as grid_spec
from sklearn.mixture import GaussianMixture

# --- Ridgeplot for distributions ---
def ridgeplot(ax, data_dict, colors, xlabel, title, show_labels=True):
    # data_dict: dict of {label: values}, colors: list of colors in legend order
    labels = list(data_dict.keys())
    n = len(labels)
    y_positions = np.arange(n)
    for i, label in enumerate(labels):
        values = data_dict[label]
        if len(values) > 1:
            x_d = np.linspace(min(values), max(values), 1000)
            kde = KernelDensity(bandwidth=(max(values)-min(values))/30 or 1e-3, kernel='gaussian')
            kde.fit(np.array(values)[:, None])
            logprob = kde.score_samples(x_d[:, None])
            y = np.exp(logprob)
            y = y / y.sum() * 100  # normalize area to 100%
            ax.plot(x_d, y + y_positions[i], color=colors[i], lw=1.5)
            ax.fill_between(x_d, y_positions[i], y + y_positions[i], color=colors[i], alpha=0.7)
        else:
            # Flat line for missing/empty data
            ax.plot([0, 1], [y_positions[i], y_positions[i]], color=colors[i], lw=1.5, alpha=0.2)
        if show_labels:
            ax.text(ax.get_xlim()[0] - 0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]), y_positions[i], label, fontweight='bold', fontsize=12, ha='right', va='center')
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

def load_bert_nodes():
    """Load nodes from BERT similarity graph to use as filter."""
    with open('Graphs/reformatted/publication_similarity_graph_bert.json', 'r') as f:
        data = json.load(f)
        return set(node['id'] for node in data['nodes'])

def load_graphs():
    """Load all graphs from the Graphs directory."""
    graphs = {}
    valid_nodes = load_bert_nodes()
    
    graph_files = [
        'Graphs/citation_net no-hop.json',  # Direct citations
        'Graphs/citation_net.json',         # One-hop citations
        'Graphs/reformatted/publication_similarity_graph_bert.json',
        'Graphs/reformatted/publication_similarity_graph_w2v.json',
        #'Graphs/paperGraph.postThresh.mean.csv.gz',
        'Graphs/normalized_graph.json'
    ]
    
    for file in graph_files:
        if file.endswith('.json'):
            with open(file, 'r') as f:
                data = json.load(f)
                G = nx.Graph()
                
                # Handle both JSON formats
                if 'nodes' in data:
                    # Format with explicit nodes list
                    G.add_nodes_from([node['id'] for node in data['nodes']])
                    
                    # Add edges with weights if available, otherwise use default weight of 1.0
                    for edge in data['edges']:
                        source = edge['source'] if 'source' in edge else edge.get('from')
                        target = edge['target'] if 'target' in edge else edge.get('to')
                        weight = edge.get('weight', 1.0)  # Default weight of 1.0 if not specified
                        if source in valid_nodes and target in valid_nodes:
                            G.add_edge(source, target, weight=weight)
                else:
                    # Format with just edges list
                    edges = data.get('edges', data)  # Handle both top-level edges and whole-file edges
                    for edge in edges:
                        source = edge.get('source', edge.get('from'))
                        target = edge.get('target', edge.get('to'))
                        weight = edge.get('weight', 1.0)  # Default weight of 1.0 if not specified
                        if source in valid_nodes and target in valid_nodes:
                            G.add_edge(source, target, weight=weight)
        
        # Filter nodes to match BERT graph nodes
        nodes_to_remove = set(G.nodes()) - valid_nodes
        G.remove_nodes_from(nodes_to_remove)
        
        graph_name = file.split('/')[-1].replace('.json', '').replace('.gexf', '')
        graphs[graph_name] = G
    
    # Load keyword similarity graph
    df = pd.read_csv('Graphs/paperGraph.postThresh.mean.csv.gz', compression='gzip')
    G_keyword = nx.from_pandas_edgelist(df, 'Source', 'Target', 'val')
    graphs['keyword_similarity'] = G_keyword
    
    return graphs

def calculate_metrics(G):
    """Calculate various graph metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Global metrics
    metrics['transitivity'] = nx.transitivity(G) if G.number_of_edges() > 0 else 0
    
    try:
        metrics['average_clustering'] = nx.average_clustering(G)
    except:
        metrics['average_clustering'] = 0
    
    # Calculate diameter and average path length for largest connected component
    if G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc)
        try:
            metrics['diameter'] = nx.diameter(G_lcc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_lcc)
        except:
            metrics['diameter'] = 0
            metrics['avg_path_length'] = 0
    else:
        metrics['diameter'] = 0
        metrics['avg_path_length'] = 0
    
    # Calculate modularity
    if G.number_of_edges() > 0:
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            metrics['modularity'] = nx.algorithms.community.modularity(G, communities)
        except:
            metrics['modularity'] = 0
    else:
        metrics['modularity'] = 0
    
    # Centrality measures
    if G.number_of_nodes() > 0:
        metrics['degree_centrality'] = list(nx.degree_centrality(G).values())
        try:
            metrics['betweenness_centrality'] = list(nx.betweenness_centrality(G).values())
        except:
            metrics['betweenness_centrality'] = [0] * G.number_of_nodes()
        try:
            metrics['clustering_coefficient'] = list(nx.clustering(G).values())
        except:
            metrics['clustering_coefficient'] = [0] * G.number_of_nodes()
    else:
        metrics['degree_centrality'] = []
        metrics['betweenness_centrality'] = []
        metrics['clustering_coefficient'] = []
    
    # Edge weight distribution
    metrics['edge_weights'] = [data['weight'] if 'weight' in data else data.get('val', 1.0) for _, _, data in G.edges(data=True)]
    
    return metrics

def normalize_edge_weights(G):
    """Normalize edge weights in a graph to [0, 1], for both 'weight' and 'val' attributes."""
    # Check which attribute is present
    edge_attr = None
    for _, _, d in G.edges(data=True):
        if 'weight' in d:
            edge_attr = 'weight'
            break
        elif 'val' in d:
            edge_attr = 'val'
            break
    if edge_attr is None:
        return G
    weights = [d.get(edge_attr, 1.0) for _, _, d in G.edges(data=True)]
    if not weights:
        return G
    min_w, max_w = min(weights), max(weights)
    for u, v, d in G.edges(data=True):
        w = d.get(edge_attr, 1.0)
        if max_w > min_w:
            d[edge_attr] = (w - min_w) / (max_w - min_w)
        else:
            d[edge_attr] = 0.0
    return G

def plot_metrics(graphs, metrics):
    """Create comparison plots for different metrics."""
    fig = plt.figure(figsize=(20, 15))
    
    # Define graph name mappings and colors (swapping Combined Graph and Keyword Similarity)
    name_mapping = {
        'citation_net no-hop': 'Direct Citations',
        'citation_net': 'One-Hop Citations',
        'publication_similarity_graph_bert': 'Abstract Similarity (BERT)',
        'publication_similarity_graph_w2v': 'Abstract Similarity (Word2Vec)',
        'normalized_graph': 'Combined Graph',
        'keyword_similarity': 'Keyword Similarity'
    }
    legend_order = [
        'Direct Citations',
        'One-Hop Citations',
        'Abstract Similarity (BERT)',
        'Abstract Similarity (Word2Vec)',
        'Keyword Similarity',
        'Combined Graph'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', '#9467bd']
    color_map = {name: color for name, color in zip(legend_order, colors)}
    
    gs = plt.GridSpec(2, 3, top=0.85, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.3)
    
    # Only use graphs present in name_mapping
    filtered_keys = [k for k in graphs if k in name_mapping]
    
    # Normalize edge weights for all graphs
    norm_graphs = {k: normalize_edge_weights(G.copy()) for k, G in graphs.items() if k in name_mapping}
    
    # Build data_dict for each metric, always in legend_order
    label_order = legend_order
    color_order = [color_map[n] for n in label_order]
    # Edge weights
    weights_dict = {}
    for name in label_order:
        if name in name_mapping.values():
            key = [k for k, v in name_mapping.items() if v == name][0]
            G = norm_graphs.get(key)
            if G is not None:
                weights = [data['weight'] if 'weight' in data else data.get('val', 1.0) for _, _, data in G.edges(data=True)]
                weights_dict[name] = weights
            else:
                weights_dict[name] = []
        else:
            weights_dict[name] = []
    ax1 = plt.subplot(gs[0, 0])
    ridgeplot(ax1, weights_dict, color_order, 'Weight', 'Edge Weight Distribution', show_labels=True)
    ax1.set_yticks([])
    # Degree
    degrees_dict = {}
    for name in label_order:
        if name in name_mapping.values():
            key = [k for k, v in name_mapping.items() if v == name][0]
            G = norm_graphs.get(key)
            if G is not None:
                degrees = [d for n, d in G.degree()]
                degrees_dict[name] = degrees
            else:
                degrees_dict[name] = []
        else:
            degrees_dict[name] = []
    ax2 = plt.subplot(gs[0, 1])
    ridgeplot(ax2, degrees_dict, color_order, 'Degree', 'Degree Distribution', show_labels=False)
    ax2.set_yticks([])
    # Betweenness
    betweenness_dict = {}
    for name in label_order:
        if name in name_mapping.values():
            key = [k for k, v in name_mapping.items() if v == name][0]
            G = norm_graphs.get(key)
            if G is not None:
                betweenness = list(nx.betweenness_centrality(G).values())
                betweenness_dict[name] = betweenness
            else:
                betweenness_dict[name] = []
        else:
            betweenness_dict[name] = []
    ax3 = plt.subplot(gs[0, 2])
    ridgeplot(ax3, betweenness_dict, color_order, 'Betweenness Centrality', 'Betweenness Centrality Distribution', show_labels=False)
    ax3.set_yticks([])
    # Clustering
    clustering_dict = {}
    for name in label_order:
        if name in name_mapping.values():
            key = [k for k, v in name_mapping.items() if v == name][0]
            G = norm_graphs.get(key)
            if G is not None:
                clustering = list(nx.clustering(G).values())
                clustering_dict[name] = clustering
            else:
                clustering_dict[name] = []
        else:
            clustering_dict[name] = []
    ax4 = plt.subplot(gs[1, 0])
    ridgeplot(ax4, clustering_dict, color_order, 'Clustering Coefficient', 'Clustering Coefficient Distribution', show_labels=True)
    ax4.set_yticks([])
    
    # 5. Number of edges comparison (Sorted bar plot)
    ax5 = plt.subplot(gs[1, 1])
    edge_counts = [(name_mapping[name], norm_graphs[name].number_of_edges()) for name in filtered_keys]
    edge_counts = sorted(edge_counts, key=lambda x: legend_order.index(x[0]))
    names, counts = zip(*edge_counts)
    bars = ax5.bar(range(len(names)), counts, color=[color_map[n] for n in names])
    ax5.set_title('Number of Edges')
    ax5.set_xticks(range(len(names)))
    ax5.set_xticklabels(names, rotation=45, ha='right')
    
    # 6. Global metrics comparison (Sorted bar plot)
    ax6 = plt.subplot(gs[1, 2])
    metrics_to_plot = ['density', 'transitivity', 'modularity']
    x = np.arange(len(metrics_to_plot))
    width = 0.13
    graph_metrics = []
    for name in filtered_keys:
        values = [metrics[name][m] for m in metrics_to_plot]
        graph_metrics.append((name_mapping[name], values))
    graph_metrics = sorted(graph_metrics, key=lambda x: legend_order.index(x[0]))
    for i, (name, values) in enumerate(graph_metrics):
        ax6.bar(x + i*width, values, width, label=name, color=color_map[name])
    ax6.set_title('Global Graph Metrics')
    ax6.set_xticks(x + width*2.5)
    ax6.set_xticklabels(['Density', 'Transitivity', 'Modularity'])
    
    # Create a single legend for the entire figure at the top with correct order
    handles = [plt.Line2D([0], [0], color=color_map[name], lw=8) for name in legend_order]
    labels = legend_order
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=3, frameon=False, columnspacing=3.0, handletextpad=1.0, fontsize=10, handlelength=1.5)
    
    # Remove individual legends if they exist
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    plt.savefig('Graphs/graph_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # After all plots, call the new plot function
    plot_diameter_avgpath(metrics, name_mapping, legend_order, color_map)

def plot_diameter_avgpath(metrics, name_mapping, legend_order, color_map):
    """Plot diameter and average path length for each graph, grouped by metric (side-by-side bars within each group)."""
    filtered_keys = [k for k in metrics if k in name_mapping]
    diameters = [metrics[k]['diameter'] for k in filtered_keys]
    avg_paths = [metrics[k]['avg_path_length'] for k in filtered_keys]
    names = [name_mapping[k] for k in filtered_keys]
    # Ensure order matches legend_order
    order = [legend_order.index(n) for n in names]
    names = [n for _, n in sorted(zip(order, names))]
    diameters = [d for _, d in sorted(zip(order, diameters))]
    avg_paths = [d for _, d in sorted(zip(order, avg_paths))]
    colors = [color_map[n] for n in names]
    n_graphs = len(names)
    x = np.arange(2)  # 0: Diameter, 1: Avg Path Length
    width = 0.8 / n_graphs
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (val_d, val_a, name, color) in enumerate(zip(diameters, avg_paths, names, colors)):
        ax.bar(x[0] + i*width - 0.4 + width/2, val_d, width, label=name if x[0]==0 else "", color=color)
        ax.bar(x[1] + i*width - 0.4 + width/2, val_a, width, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(['Diameter', 'Avg Path Length'])
    ax.set_ylabel('Value')
    ax.set_title('Global Graph Metrics')
    handles = [plt.Line2D([0], [0], color=color_map[n], lw=8) for n in names]
    ax.legend(handles, names, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('Graphs/diameter_avgpath.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_metrics_table(metrics):
    """Print a table of metrics for all graphs."""
    print("\nGraph Metrics Comparison:")
    print("-" * 100)
    print(f"{'Metric':<20}", end="")
    for graph_name in metrics.keys():
        print(f"{graph_name:<15}", end="")
    print()
    print("-" * 100)
    
    metric_names = ['num_nodes', 'num_edges', 'density', 'diameter', 
                   'avg_path_length', 'transitivity', 'modularity']
    
    for metric in metric_names:
        print(f"{metric:<20}", end="")
        for graph_name in metrics.keys():
            value = metrics[graph_name][metric]
            if isinstance(value, float):
                print(f"{value:<15.4f}", end="")
            else:
                print(f"{value:<15}", end="")
        print()
    print("-" * 100)

def compute_node_features(G):
    """Compute node features for clustering."""
    features = []
    
    # Compute various centrality measures
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    close_cent = nx.closeness_centrality(G)
    clustering_coef = nx.clustering(G)
    
    # Create feature matrix
    for node in G.nodes():
        node_features = [
            degree_cent[node],
            between_cent[node],
            close_cent[node],
            clustering_coef[node]
        ]
        features.append(node_features)
    
    return np.array(features)

def compute_elbo(X, n_components_range):
    """Compute ELBO for different numbers of clusters using Gaussian Mixture Model."""
    elbo_scores = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        elbo_scores.append(gmm.lower_bound_)
    return elbo_scores

def plot_elbo_selection(X, graph_name, n_components_range=range(2, 11), optimal_n=None):
    """Plot ELBO scores for different numbers of clusters."""
    elbo_scores = compute_elbo(X, n_components_range)
    
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, elbo_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('ELBO Score')
    plt.title(f'ELBO Selection for {graph_name}')
    plt.grid(True)
    
    # Use provided optimal_n if specified, otherwise use maximum
    if optimal_n is None:
        optimal_n = n_components_range[np.argmax(elbo_scores)]
    
    # Add vertical line at optimal_n
    plt.axvline(x=optimal_n, color='r', linestyle='--', 
                label=f'Optimal: {optimal_n}')
    plt.legend()
    
    # Save the plot
    plt.savefig(f'Graphs/elbo_selection_{graph_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_n, elbo_scores

def perform_clustering_analysis(graphs, n_clusters_range=range(2, 11)):
    """Perform clustering analysis using multiple methods with ELBO-based cluster selection."""
    clustering_results = {}
    
    # Define optimal number of clusters for each graph
    optimal_clusters = {
        'citation_net no-hop': 6,
        'citation_net': 7,
        'publication_similarity_graph_bert': 9,
        'publication_similarity_graph_w2v': 8,
        'normalized_graph': 9,
        'keyword_similarity': 7
    }
    
    # Define resolution parameters for Louvain (lower values = larger communities)
    louvain_resolutions = {
        'citation_net no-hop': 0.6,
        'citation_net': 0.6,
        'publication_similarity_graph_bert': 0.6,
        'publication_similarity_graph_w2v': 0.6,
        'normalized_graph': 0.6,
        'keyword_similarity': 0.6
    }
    
    # Define the order of graphs for plotting
    graph_order = [
        'citation_net no-hop',
        'citation_net',
        'publication_similarity_graph_bert',
        'publication_similarity_graph_w2v',
        'keyword_similarity',
        'normalized_graph'
    ]
    
    # Process graphs in the specified order
    for name in graph_order:
        G = graphs[name]
        # Get node features
        features = compute_node_features(G)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Compute ELBO scores with specified optimal number of clusters
        optimal_n, elbo_scores = plot_elbo_selection(
            features_scaled, 
            name, 
            n_clusters_range,
            optimal_n=optimal_clusters.get(name)
        )
        
        # Store results for this graph
        clustering_results[name] = {
            'features': features_scaled,
            'methods': {},
            'elbo_scores': elbo_scores,
            'optimal_n': optimal_n
        }
        
        # Get list of nodes for mapping
        nodes = list(G.nodes())
        
        # 1. Spectral Clustering
        try:
            spectral = SpectralClustering(n_clusters=optimal_n, 
                                        affinity='nearest_neighbors')
            spectral_labels = spectral.fit_predict(features_scaled)
        except:
            spectral_labels = np.zeros(len(nodes))
        
        # 2. K-means
        try:
            kmeans = KMeans(n_clusters=optimal_n)
            kmeans_labels = kmeans.fit_predict(features_scaled)
        except:
            kmeans_labels = np.zeros(len(nodes))
        
        # 3. Louvain Community Detection with resolution parameter
        try:
            # Create a copy of the graph without singleton nodes
            G_louvain = G.copy()
            singleton_nodes = [node for node in G_louvain.nodes() if G_louvain.degree(node) == 0]
            G_louvain.remove_nodes_from(singleton_nodes)
            
            # Perform Louvain clustering on the graph without singletons
            louvain_partition = community_louvain.best_partition(
                G_louvain, 
                resolution=louvain_resolutions.get(name, 0.6)
            )
            
            # Create labels array for all nodes
            louvain_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in louvain_partition:
                    louvain_labels[i] = louvain_partition[node]
                else:
                    # Assign singleton nodes to their own cluster
                    louvain_labels[i] = max(louvain_partition.values()) + 1 if louvain_partition else 0
        except:
            louvain_labels = np.zeros(len(nodes))
        
        # Store clustering results
        methods = {
            'Spectral': spectral_labels,
            'K-means': kmeans_labels,
            'Louvain': louvain_labels
        }
        
        # Compute metrics for each method
        for method_name, labels in methods.items():
            # Compute silhouette score if more than one cluster
            n_clusters = len(np.unique(labels))
            if n_clusters > 1:
                try:
                    silhouette = silhouette_score(features_scaled, labels)
                except:
                    silhouette = 0
            else:
                silhouette = 0
            
            # Compute modularity
            try:
                communities = []
                for label in np.unique(labels):
                    community = []
                    for idx, l in enumerate(labels):
                        if l == label:
                            community.append(nodes[idx])
                    communities.append(community)
                
                modularity = nx.community.modularity(G, communities)
            except:
                modularity = 0
            
            # Store results
            clustering_results[name]['methods'][method_name] = {
                'labels': labels,
                'silhouette': silhouette,
                'modularity': modularity,
                'n_clusters': n_clusters
            }
    
    return clustering_results

def plot_clustering_results(graphs, clustering_results):
    """Create visualization of clustering results with ELBO plot."""
    n_graphs = len(graphs)
    n_methods = len(next(iter(clustering_results.values()))['methods'])
    
    # Set up the figure
    fig = plt.figure(figsize=(25, 6 * n_graphs))
    gs = plt.GridSpec(n_graphs, n_methods + 2, width_ratios=[1, 1, 1, 0.5, 0.5])
    
    # Color palette for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Plot each graph's clustering results
    for i, (graph_name, results) in enumerate(clustering_results.items()):
        # Compute t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(results['features'])
        
        # Plot each clustering method
        for j, (method_name, method_results) in enumerate(results['methods'].items()):
            ax = plt.subplot(gs[i, j])
            
            # Scatter plot with cluster colors
            labels = method_results['labels']
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[int(label) % 20]], label=f'Cluster {int(label)}',
                          alpha=0.6, s=50)
            
            # Add metrics as text
            metrics_text = f"Silhouette: {method_results['silhouette']:.3f}\n"
            metrics_text += f"Modularity: {method_results['modularity']:.3f}\n"
            metrics_text += f"Clusters: {method_results['n_clusters']}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title(f"{graph_name}\n{method_name} Clustering")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add metrics comparison bar plot
        ax_metrics = plt.subplot(gs[i, -2])
        methods = list(results['methods'].keys())
        silhouette_scores = [results['methods'][m]['silhouette'] for m in methods]
        modularity_scores = [results['methods'][m]['modularity'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax_metrics.bar(x - width/2, silhouette_scores, width, label='Silhouette')
        ax_metrics.bar(x + width/2, modularity_scores, width, label='Modularity')
        
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(methods, rotation=45)
        ax_metrics.set_title('Clustering Metrics')
        ax_metrics.legend()
        
        # Add ELBO plot
        ax_elbo = plt.subplot(gs[i, -1])
        n_components = range(2, len(results['elbo_scores']) + 2)
        ax_elbo.plot(n_components, results['elbo_scores'], 'bo-')
        ax_elbo.axvline(x=results['optimal_n'], color='r', linestyle='--', 
                       label=f'Optimal: {results["optimal_n"]}')
        ax_elbo.set_xlabel('Number of Clusters')
        ax_elbo.set_ylabel('ELBO Score')
        ax_elbo.set_title('ELBO Selection')
        ax_elbo.grid(True)
        ax_elbo.legend()
    
    plt.tight_layout()
    plt.savefig('Graphs/clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load all graphs
    print("Loading graphs...")
    graphs = load_graphs()
    
    # Calculate metrics for each graph
    print("Calculating metrics...")
    all_metrics = {}
    for name, G in tqdm(graphs.items()):
        all_metrics[name] = calculate_metrics(G)
    
    # Create plots
    print("Creating plots...")
    plot_metrics(graphs, all_metrics)
    
    # Perform clustering analysis
    print("Performing clustering analysis...")
    clustering_results = perform_clustering_analysis(graphs)
    
    # Create clustering visualization
    print("Creating clustering visualization...")
    plot_clustering_results(graphs, clustering_results)
    
    # Print metrics table
    print_metrics_table(all_metrics)
    
    print("\nAnalysis complete! Results saved to:")
    print("- 'Graphs/graph_comparison.png'")
    print("- 'Graphs/clustering_comparison.png'")

if __name__ == "__main__":
    main() 