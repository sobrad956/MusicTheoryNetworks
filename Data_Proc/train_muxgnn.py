import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, GCNConv
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage

# Add BaseStorage to safe globals
add_safe_globals([BaseStorage])

class MuxGNN(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout=0.2,
            activation='elu'
    ):
        super(MuxGNN, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.dropout = dropout
        self.activation = activation.casefold()
        
        # Create GNN layers for each relation type
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            layer_dict = nn.ModuleDict()
            for rel in relations:
                if gnn_type == 'gcn':
                    layer_dict[rel] = GCNConv(feat_dim if _ == 0 else embed_dim, embed_dim)
                elif gnn_type == 'gat':
                    layer_dict[rel] = GATConv(feat_dim if _ == 0 else embed_dim, embed_dim)
                elif gnn_type == 'gin':
                    nn_layer = nn.Sequential(
                        nn.Linear(feat_dim if _ == 0 else embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, embed_dim)
                    )
                    layer_dict[rel] = GINConv(nn_layer)
            self.gnn_layers.append(layer_dict)
        
        # Semantic attention layer
        self.semantic_attention = SemanticAttention(num_relations=len(relations), in_dim=embed_dim, dim_a=dim_a)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data):
        # Initial node features
        x = data['node'].x
        
        # Process each layer
        for layer in self.gnn_layers:
            # Process each relation type
            relation_embeddings = []
            for rel in self.relations:
                edge_index = data['node', rel, 'node'].edge_index
                if edge_index is not None:
                    rel_emb = layer[rel](x, edge_index)
                    if self.activation == 'elu':
                        rel_emb = F.elu(rel_emb)
                    elif self.activation == 'relu':
                        rel_emb = F.relu(rel_emb)
                    relation_embeddings.append(rel_emb)
                else:
                    relation_embeddings.append(torch.zeros_like(x))
            
            # Stack relation embeddings
            relation_embeddings = torch.stack(relation_embeddings, dim=1)  # [num_nodes, num_relations, embed_dim]
            
            # Apply semantic attention
            x = self.semantic_attention(relation_embeddings)  # [num_nodes, embed_dim]
            x = self.dropout_layer(x)
        
        return x

class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.dim_a = dim_a
        
        self.W = nn.Parameter(torch.zeros(size=(in_dim, dim_a)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.b = nn.Parameter(torch.zeros(size=(1, dim_a)))
        
        self.q = nn.Parameter(torch.zeros(size=(dim_a, 1)))
        nn.init.xavier_uniform_(self.q.data)
    
    def forward(self, h):
        # h shape: [num_nodes, num_relations, in_dim]
        
        # Calculate attention
        w = torch.tanh(torch.matmul(h, self.W) + self.b)  # [num_nodes, num_relations, dim_a]
        w = torch.matmul(w, self.q)  # [num_nodes, num_relations, 1]
        
        alpha = F.softmax(w, dim=1)  # [num_nodes, num_relations, 1]
        
        # Apply attention
        out = torch.sum(alpha * h, dim=1)  # [num_nodes, in_dim]
        
        return out

def train_model(
        model,
        data,
        val_edges,
        test_edges,
        device,
        epochs=100,
        batch_size=64,
        lr=0.001,
        patience=10
):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Get node embeddings
        node_embeddings = model(data)
        
        # Calculate loss using positive and negative edges
        total_loss = 0
        for rel in val_edges:
            pos_src, pos_dst, labels = val_edges[rel]
            pos_src = torch.tensor(pos_src, dtype=torch.long, device=device)
            pos_dst = torch.tensor(pos_dst, dtype=torch.long, device=device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)
            
            # Get embeddings for source and destination nodes
            src_emb = node_embeddings[pos_src]
            dst_emb = node_embeddings[pos_dst]
            
            # Calculate similarity scores
            scores = torch.sum(src_emb * dst_emb, dim=1)
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            node_embeddings = model(data)
            val_aucs = []
            val_aps = []
            
            for rel in val_edges:
                pos_src, pos_dst, labels = val_edges[rel]
                pos_src = torch.tensor(pos_src, dtype=torch.long, device=device)
                pos_dst = torch.tensor(pos_dst, dtype=torch.long, device=device)
                
                # Get embeddings for source and destination nodes
                src_emb = node_embeddings[pos_src]
                dst_emb = node_embeddings[pos_dst]
                
                # Calculate similarity scores
                scores = torch.sum(src_emb * dst_emb, dim=1).cpu().numpy()
                
                # Calculate metrics
                val_aucs.append(roc_auc_score(labels, scores))
                val_aps.append(average_precision_score(labels, scores))
            
            val_auc = np.mean(val_aucs)
            val_ap = np.mean(val_aps)
            
            print(f'Epoch {epoch:03d}, Loss: {total_loss.item():.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
            
            # Early stopping
            if val_auc > best_val_score:
                best_val_score = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping!')
                    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data)
        test_aucs = []
        test_aps = []
        
        for rel in test_edges:
            pos_src, pos_dst, labels = test_edges[rel]
            pos_src = torch.tensor(pos_src, dtype=torch.long, device=device)
            pos_dst = torch.tensor(pos_dst, dtype=torch.long, device=device)
            
            # Get embeddings for source and destination nodes
            src_emb = node_embeddings[pos_src]
            dst_emb = node_embeddings[pos_dst]
            
            # Calculate similarity scores
            scores = torch.sum(src_emb * dst_emb, dim=1).cpu().numpy()
            
            # Calculate metrics
            test_aucs.append(roc_auc_score(labels, scores))
            test_aps.append(average_precision_score(labels, scores))
        
        test_auc = np.mean(test_aucs)
        test_ap = np.mean(test_aps)
        
        print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
    
    return model

def main():
    # Load data
    data = torch.load('data/publication_similarity/train_G.pt', weights_only=False)
    
    with open('data/publication_similarity/val_edges.json', 'r') as f:
        val_edges = json.load(f)
    
    with open('data/publication_similarity/test_edges.json', 'r') as f:
        test_edges = json.load(f)
    
    # Model parameters
    gnn_type = 'gin'  # 'gcn', 'gat', or 'gin'
    num_gnn_layers = 2
    relations = ['bert_similarity', 'word2vec_similarity']
    feat_dim = data['node'].x.size(1)
    embed_dim = 128
    dim_a = 32
    dropout = 0.2
    activation = 'elu'
    
    # Create model
    model = MuxGNN(
        gnn_type=gnn_type,
        num_gnn_layers=num_gnn_layers,
        relations=relations,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        dim_a=dim_a,
        dropout=dropout,
        activation=activation
    )
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 64
    lr = 0.001
    patience = 10
    
    # Train model
    model = train_model(
        model=model,
        data=data,
        val_edges=val_edges,
        test_edges=test_edges,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience
    )
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/muxgnn.pt')

if __name__ == '__main__':
    main() 