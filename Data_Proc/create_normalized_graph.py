import json
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import HeteroData

def load_multiplex_network():
    """Load the multiplex network and node mappings."""
    # Load the multiplex network
    torch.serialization.add_safe_globals([HeteroData])
    data = torch.load('Graphs/multiplex_network.pt', weights_only=False)
    
    # Load node mappings
    with open('Graphs/node_mappings.json', 'r') as f:
        mappings = json.load(f)
        idx_to_node = mappings['idx_to_node']
    
    return data, idx_to_node

def normalize_weights(edge_index, edge_type, category):
    """Normalize edge weights for a given edge type."""
    # Get the number of edges
    num_edges = edge_index.shape[1]
    
    # Create weights based on edge type
    if category == 'citation':
        # For citation networks, use 1.0 for direct citations and 0.5 for one-hop citations
        if edge_type == 'direct_citation':
            weights = torch.ones(num_edges)
        else:  # one_hop_citation
            weights = torch.full((num_edges,), 0.5)
    else:  # similarity
        # For similarity networks, use degree-based weights
        # Higher degree nodes get lower weights to avoid bias towards hubs
        degrees = torch.bincount(edge_index.flatten())
        max_degree = degrees.max().item()
        source_degrees = degrees[edge_index[0]]
        target_degrees = degrees[edge_index[1]]
        # Weight is inversely proportional to the average degree of source and target
        weights = 1.0 - 0.5 * (source_degrees + target_degrees).float() / max_degree
    
    # Normalize weights to [0, 1]
    min_weight = weights.min()
    max_weight = weights.max()
    if max_weight > min_weight:
        normalized_weights = (weights - min_weight) / (max_weight - min_weight)
    else:
        normalized_weights = weights
    
    return normalized_weights

def create_normalized_graph():
    """Create a normalized undirected graph from the multiplex network."""
    # Load the multiplex network
    data, idx_to_node = load_multiplex_network()
    
    # Print available edge types
    print("\nAvailable edge types:")
    for key in data.edge_index_dict.keys():
        print(f"{key}: {data.edge_index_dict[key].shape}")
    
    # Initialize edge weights dictionary
    edge_weights = defaultdict(list)
    
    # Process each edge type
    edge_types = [
        ('one_hop_citation', 'citation'),
        ('direct_citation', 'citation'),
        ('bert', 'similarity'),
        ('w2v', 'similarity'),
        ('keywords', 'similarity')
    ]
    
    for edge_type, category in edge_types:
        edge_key = ('node', edge_type, 'node')
        if edge_key in data.edge_index_dict:
            edge_index = data.edge_index_dict[edge_key]
            weights = normalize_weights(edge_index, edge_type, category)
            
            print(f"\nProcessing {edge_type}:")
            print(f"Edge index shape: {edge_index.shape}")
            print(f"Number of edges: {edge_index.shape[1]}")
            print(f"Weight range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
            print(f"Average weight: {weights.mean().item():.4f}")
            
            # Add weights to the dictionary
            for i in range(edge_index.shape[1]):
                source_idx = str(edge_index[0, i].item())
                target_idx = str(edge_index[1, i].item())
                
                if source_idx in idx_to_node and target_idx in idx_to_node:
                    source = idx_to_node[source_idx]
                    target = idx_to_node[target_idx]
                    edge_weights[(source, target)].append(weights[i].item())
                    edge_weights[(target, source)].append(weights[i].item())  # Make undirected
                else:
                    print(f"Warning: Node indices {source_idx}, {target_idx} not found in mapping")
        else:
            print(f"\nWarning: Edge type {edge_key} not found in multiplex network")
    
    # Create the normalized graph
    normalized_graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": node_id, "mto_paper": True} for node_id in idx_to_node.values()],
        "edges": []
    }
    
    # Add edges with averaged weights
    seen_edges = set()  # To avoid duplicate edges
    for (source, target), weights in edge_weights.items():
        # Only add each edge once (since it's undirected)
        edge_key = tuple(sorted([source, target]))
        if edge_key not in seen_edges:
            avg_weight = np.mean(weights)
            normalized_graph["edges"].append({
                "source": source,
                "target": target,
                "weight": avg_weight
            })
            seen_edges.add(edge_key)
    
    # Save the normalized graph
    with open('Graphs/normalized_graph.json', 'w') as f:
        json.dump(normalized_graph, f, indent=4)
    
    # Print statistics
    print("\nNormalized graph statistics:")
    print(f"Number of nodes: {len(normalized_graph['nodes'])}")
    print(f"Number of edges: {len(normalized_graph['edges'])}")
    
    if normalized_graph['edges']:
        weights = [edge['weight'] for edge in normalized_graph['edges']]
        print(f"Average edge weight: {np.mean(weights):.4f}")
        print(f"Min edge weight: {min(weights):.4f}")
        print(f"Max edge weight: {max(weights):.4f}")
        
        # Print weight distribution
        print("\nWeight distribution:")
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(weights, bins=bins)
        for i in range(len(hist)):
            print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} edges")
    else:
        print("Warning: No edges found in the normalized graph")

if __name__ == '__main__':
    create_normalized_graph() 