import argparse
import logging
import os
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import dgl

from muxGNN.model.mux_gnn import MuxGNN
from muxGNN.model.sample import NegativeSamplingLoss, NeighborSampler
import muxGNN.utils as utils

def get_edge_type_combinations():
    """Define all edge type combinations to test."""
    all_types = ['direct_citation', 'one_hop_citation', 'bert', 'w2v', 'keywords']
    combinations = {
        'all': all_types,
        'citation_onehop': ['direct_citation', 'one_hop_citation'],
        'bert_w2v': ['bert', 'w2v'],
        'keywords': ['keywords'],
        'citation_onehop_keywords': ['direct_citation', 'one_hop_citation', 'keywords'],
        'citation_onehop_bert_w2v': ['direct_citation', 'one_hop_citation', 'bert', 'w2v'],
        'bert_w2v_keywords': ['bert', 'w2v', 'keywords']
    }
    return combinations

def prepare_data(multiplex_network_path, selected_edge_types=None):
    """Load and prepare the multiplex network for link prediction."""
    # Load the multiplex network
    hetero_data = torch.load(multiplex_network_path)
    
    # Convert PyTorch Geometric HeteroData to DGL graph
    edge_dict = {}
    available_edge_types = set()
    
    # First, collect all available edge types
    for edge_type in hetero_data.edge_types:
        src_type, rel_type, dst_type = edge_type
        available_edge_types.add(rel_type)
    
    # Filter selected edge types to only include available ones
    if selected_edge_types is not None:
        selected_edge_types = [et for et in selected_edge_types if et in available_edge_types]
        if not selected_edge_types:
            raise ValueError(f"None of the selected edge types {selected_edge_types} are available in the graph. Available types: {available_edge_types}")
    
    # Create edge dictionary with available edge types
    for edge_type in hetero_data.edge_types:
        src_type, rel_type, dst_type = edge_type
        if selected_edge_types is None or rel_type in selected_edge_types:
            edge_dict[(src_type, rel_type, dst_type)] = (
                hetero_data[edge_type].edge_index[0],
                hetero_data[edge_type].edge_index[1]
            )
    
    train_G = dgl.heterograph(edge_dict)
    
    # Add node features if they exist
    if hasattr(hetero_data['node'], 'x'):
        train_G.ndata['feat'] = hetero_data['node'].x
    else:
        # Create random features if none exist
        train_G.ndata['feat'] = torch.randn(train_G.number_of_nodes(), 64)
    
    # Create validation and test edge sets
    val_edges = {}
    test_edges = {}
    
    # For each edge type, hold out 10% for validation and 10% for testing
    for etype in train_G.canonical_etypes:
        edges = train_G.edges(etype=etype)
        num_edges = len(edges[0])
        
        # Create indices for shuffling
        indices = torch.randperm(num_edges)
        val_size = int(0.1 * num_edges)
        test_size = int(0.1 * num_edges)
        
        # Split indices
        val_indices = indices[:val_size]
        test_indices = indices[val_size:val_size + test_size]
        train_indices = indices[val_size + test_size:]
        
        # Create validation edge set
        val_edges[etype[1]] = (
            edges[0][val_indices].tolist(),
            edges[1][val_indices].tolist(),
            [1] * val_size  # Positive labels
        )
        
        # Create test edge set
        test_edges[etype[1]] = (
            edges[0][test_indices].tolist(),
            edges[1][test_indices].tolist(),
            [1] * test_size  # Positive labels
        )
        
        # Remove validation and test edges from training graph
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[val_indices] = False
        mask[test_indices] = False
        train_G = dgl.remove_edges(train_G, 
                                 torch.arange(num_edges)[~mask], 
                                 etype=etype)
        
        # Generate negative samples for validation and test sets
        num_nodes = train_G.number_of_nodes()
        for edge_set in [val_edges[etype[1]], test_edges[etype[1]]]:
            neg_src = torch.randint(0, num_nodes, (len(edge_set[0]),)).tolist()
            neg_dst = torch.randint(0, num_nodes, (len(edge_set[0]),)).tolist()
            edge_set[0].extend(neg_src)
            edge_set[1].extend(neg_dst)
            edge_set[2].extend([0] * len(neg_src))  # Negative labels
    
    return train_G, val_edges, test_edges

def run_experiment(args, edge_types, device):
    """Run a single experiment with specified edge types."""
    print(f"\nRunning experiment with edge types: {edge_types}")
    
    # Load and prepare data
    train_G, val_edges, test_edges = prepare_data('Graphs/multiplex_network.pt', edge_types)
    
    # Initialize model and training components
    feat_dim = train_G.ndata['feat'].shape[-1]
    fanouts = args.neigh_samples * args.num_layers if len(args.neigh_samples) == 1 else args.neigh_samples
    
    model = MuxGNN(
        gnn_type=args.gnn,
        num_gnn_layers=args.num_layers,
        relations=train_G.etypes,
        feat_dim=feat_dim,
        embed_dim=args.embed_dim,
        dim_a=args.dim_a,
        dropout=args.dropout,
        activation=args.activation,
    )
    
    neigh_sampler = NeighborSampler(fanouts)
    nsloss = NegativeSamplingLoss(
        train_G, 
        num_neg_samples=args.neg_samples, 
        embedding_dim=args.embed_dim, 
        dist='uniform'
    )
    
    # Train model
    val_aucs, val_f1s, val_prs = model.train_model(
        train_G=train_G,
        val_edges=val_edges,
        neigh_sampler=neigh_sampler,
        loss_module=nsloss,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
        EPOCHS=10,
        patience_limit=args.patience,
        num_workers=args.num_workers,
        device=device,
        model_dir='saved_models'
    )
    
    # Evaluate on test set
    test_aucs, test_f1s, test_prs = model.eval_model(
        train_G,
        test_edges,
        neigh_sampler,
        batch_size=args.batch_size,
        device=device
    )
    
    return {
        'val_aucs': val_aucs,
        'val_f1s': val_f1s,
        'val_prs': val_prs,
        'test_aucs': test_aucs,
        'test_f1s': test_f1s,
        'test_prs': test_prs,
        'epoch_metrics': {
            'val_aucs': val_aucs,
            'val_f1s': val_f1s,
            'val_prs': val_prs
        }
    }

def generate_results_table(results):
    """Generate a table of results and save it to CSV."""
    # Prepare data for the table
    table_data = []
    for exp_name, result in results.items():
        row = {
            'Experiment': exp_name,
            'Edge Types': ', '.join(result['edge_types']),
            'Mean Test ROC-AUC': f"{np.mean(result['test_aucs']):.4f} ± {np.std(result['test_aucs']):.4f}",
            'Mean Test F1': f"{np.mean(result['test_f1s']):.4f} ± {np.std(result['test_f1s']):.4f}",
            'Mean Test PR-AUC': f"{np.mean(result['test_prs']):.4f} ± {np.std(result['test_prs']):.4f}",
            'Mean Val ROC-AUC': f"{np.mean(result['val_aucs']):.4f} ± {np.std(result['val_aucs']):.4f}",
            'Mean Val F1': f"{np.mean(result['val_f1s']):.4f} ± {np.std(result['val_f1s']):.4f}",
            'Mean Val PR-AUC': f"{np.mean(result['val_prs']):.4f} ± {np.std(result['val_prs']):.4f}",
            'Best Epoch': np.argmax(result['val_aucs']) + 1,
            'Best Val ROC-AUC': f"{np.max(result['val_aucs']):.4f}",
            'Best Val F1': f"{np.max(result['val_f1s']):.4f}",
            'Best Val PR-AUC': f"{np.max(result['val_prs']):.4f}"
        }
        table_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_data)
    df.to_csv('experiment_results_table.csv', index=False)
    
    # Also save as LaTeX table
    latex_table = df.to_latex(index=False, float_format=lambda x: f'{x:.4f}')
    with open('experiment_results_table.tex', 'w') as f:
        f.write(latex_table)
    
    return df

def plot_results(results, metric='test_aucs'):
    """Plot the results of all experiments."""
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    data = []
    for exp_name, result in results.items():
        mean_score = np.mean(result[metric])
        data.append({
            'Experiment': exp_name,
            'Mean Score': mean_score,
            'Edge Types': len(result['edge_types'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create bar plot
    sns.barplot(data=df, x='Experiment', y='Mean Score')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Mean {metric.replace("_", " ").title()} Across Experiments')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'experiment_results_{metric}.png')
    plt.close()

def plot_learning_curves(results):
    """Plot learning curves for all experiments."""
    metrics = ['val_aucs', 'val_f1s', 'val_prs']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for exp_name, result in results.items():
            epochs = range(1, len(result['epoch_metrics'][metric]) + 1)
            ax.plot(epochs, result['epoch_metrics'][metric], 
                   label=exp_name, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Learning Curves - {metric.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN layer to use with muxGNN. "gcn", "gat", or "gin". Default is "gin".')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of k-hop neighbor aggregations to perform.')
    parser.add_argument('--neigh-samples', type=int, default=[10], nargs='+',
                        help='Number of neighbors to sample for aggregation.')
    parser.add_argument('--embed-dim', type=int, default=200,
                        help='Size of output embedding dimension.')
    parser.add_argument('--dim-a', type=int, default=20,
                        help='Dimension of attention.')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate during training.')
    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of random walks to perform.')
    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of random walks.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Size of sliding window.')
    parser.add_argument('--neg-samples', type=int, default=5,
                        help='Number of negative samples.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size during training')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait for improvement before early stopping.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes.')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename='logs/link_prediction.log',
        filemode='a'
    )
    logging.info(args)
    
    # Get edge type combinations
    combinations = get_edge_type_combinations()
    
    # Run experiments
    results = {}
    for exp_name, edge_types in combinations.items():
        print(f"\nRunning experiment: {exp_name}")
        result = run_experiment(args, edge_types, device)
        result['edge_types'] = edge_types
        results[exp_name] = result
        
        # Log results
        logging.info(f'\nResults for {exp_name}:')
        logging.info(f'Edge types: {edge_types}')
        logging.info(f'Mean ROC-AUC: {np.mean(result["test_aucs"]):.4f}')
        logging.info(f'Mean F1: {np.mean(result["test_f1s"]):.4f}')
        logging.info(f'Mean PR-AUC: {np.mean(result["test_prs"]):.4f}')
    
    # Generate and display results table
    results_table = generate_results_table(results)
    print("\nResults Table:")
    print(results_table.to_string(index=False))
    
    # Plot results
    plot_results(results, 'test_aucs')
    plot_results(results, 'test_f1s')
    plot_results(results, 'test_prs')
    
    # Plot learning curves
    plot_learning_curves(results)

if __name__ == '__main__':
    main() 