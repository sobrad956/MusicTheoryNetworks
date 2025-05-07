import argparse
import logging
import os
from time import time
import json

import numpy as np
import torch
import dgl

from muxGNN.model.mux_gnn import MuxGNN
from muxGNN.model.sample import NegativeSamplingLoss, NeighborSampler
import muxGNN.utils as utils

def prepare_data(multiplex_network_path):
    """Load and prepare the multiplex network for link prediction."""
    # Load the multiplex network
    train_G = torch.load(multiplex_network_path)
    
    # Create validation and test edge sets
    val_edges = {}
    test_edges = {}
    
    # For each edge type, hold out 10% for validation and 10% for testing
    for etype in train_G.etypes:
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
        val_edges[etype] = (
            edges[0][val_indices].tolist(),
            edges[1][val_indices].tolist(),
            [1] * val_size  # Positive labels
        )
        
        # Create test edge set
        test_edges[etype] = (
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
        for edge_set in [val_edges[etype], test_edges[etype]]:
            neg_src = torch.randint(0, num_nodes, (len(edge_set[0]),)).tolist()
            neg_dst = torch.randint(0, num_nodes, (len(edge_set[0]),)).tolist()
            edge_set[0].extend(neg_src)
            edge_set[1].extend(neg_dst)
            edge_set[2].extend([0] * len(neg_src))  # Negative labels
    
    return train_G, val_edges, test_edges

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
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum limit on training epochs.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait for improvement before early stopping.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes.')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename='logs/link_prediction.log',
        filemode='a'
    )
    logging.info(args)
    
    # Load and prepare data
    print("Loading and preparing data...")
    start = time()
    train_G, val_edges, test_edges = prepare_data('Graphs/multiplex_network.pt')
    end = time()
    logging.info(f'Loading graph data... {end - start:.2f}s')
    
    # Initialize model and training components
    print("Initializing model...")
    start = time()
    feat_dim = train_G.ndata['feat'].shape[-1] if 'feat' in train_G.ndata else 64
    
    # If no node features exist, create random features
    if 'feat' not in train_G.ndata:
        train_G.ndata['feat'] = torch.randn(train_G.number_of_nodes(), feat_dim)
    
    fanouts = args.neigh_samples * args.num_layers if len(args.neigh_samples) == 1 else args.neigh_samples
    assert len(fanouts) == args.num_layers
    
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
    end = time()
    logging.info(f'Initializing model... {end - start:.2f}s')
    
    # Train model
    print("Training model...")
    val_aucs, val_f1s, val_prs = model.train_model(
        train_G=train_G,
        val_edges=val_edges,
        neigh_sampler=neigh_sampler,
        loss_module=nsloss,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
        EPOCHS=args.epochs,
        patience_limit=args.patience,
        num_workers=args.num_workers,
        device=device,
        model_dir='saved_models'
    )
    
    # Log validation results
    logging.info('-' * 25)
    logging.info('Validation metrics:')
    logging.info(f'ROC-AUCs: {val_aucs}')
    logging.info(f'F1s: {val_f1s}')
    logging.info(f'PR-AUCs: {val_prs}')
    
    # Evaluate on test set
    print("Evaluating model...")
    test_aucs, test_f1s, test_prs = model.eval_model(
        train_G,
        test_edges,
        neigh_sampler,
        batch_size=args.batch_size,
        device=device
    )
    
    # Log test results
    logging.info('-' * 25)
    logging.info('Test metrics:')
    logging.info(f'Mean ROC-AUC: {np.mean(test_aucs):.4f}')
    logging.info(f'ROC-AUCs: {test_aucs}')
    logging.info(f'Mean F1: {np.mean(test_f1s):.4f}')
    logging.info(f'F1s: {test_f1s}')
    logging.info(f'Mean PR-AUC: {np.mean(test_prs):.4f}')
    logging.info(f'PR-AUCs: {test_prs}')
    
    # Print results
    print("\nTest Results:")
    print(f"Mean ROC-AUC: {np.mean(test_aucs):.4f}")
    print(f"Mean F1: {np.mean(test_f1s):.4f}")
    print(f"Mean PR-AUC: {np.mean(test_prs):.4f}")

if __name__ == '__main__':
    main() 