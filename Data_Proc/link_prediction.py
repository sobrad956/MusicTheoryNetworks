import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, SAGEConv
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

class MuxGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.num_relations = num_relations
        
        # Create GNN layers for each relation
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels) for _ in range(num_relations)
        ])
        
        # Final layer for link prediction
        self.final_conv = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_indices):
        # Process each relation type
        xs = []
        for i, edge_index in enumerate(edge_indices):
            x_i = self.convs[i](x, edge_index)
            x_i = F.relu(x_i)
            xs.append(x_i)
        
        # Combine representations from different relations
        x = torch.stack(xs).mean(dim=0)
        
        # Final transformation
        x = self.final_conv(x, edge_indices[0])  # Use first relation's edges for final conv
        return x

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.gnn = MuxGNN(in_channels, hidden_channels, out_channels, num_relations)
        
    def encode(self, x, edge_indices):
        return self.gnn(x, edge_indices)
    
    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)
    
    def forward(self, x, edge_indices, edge_label_index):
        z = self.encode(x, edge_indices)
        return self.decode(z, edge_label_index)

def load_graphs(gexf_paths):
    """Load multiple graphs and create a multiplex network."""
    # Load all graphs
    graphs = [nx.read_gexf(path) for path in gexf_paths]
    
    # Create a mapping from node IDs to integers
    all_nodes = set()
    for G in graphs:
        all_nodes.update(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Convert edges to integer indices for each graph
    edge_indices = []
    for G in graphs:
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges).t().contiguous()
        edge_indices.append(edge_index)
    
    # Create node features (using node degrees as features)
    x = torch.zeros(len(all_nodes), len(graphs), dtype=torch.float)
    for i, G in enumerate(graphs):
        for node in G.nodes():
            idx = node_to_idx[node]
            x[idx, i] = G.degree(node)
    
    # Create HeteroData object
    data = HeteroData()
    data['node'].x = x
    for i, edge_index in enumerate(edge_indices):
        data['node', f'relation_{i}', 'node'].edge_index = edge_index
    
    # Store the mapping
    data.node_to_idx = node_to_idx
    data.idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    return data

def generate_negative_edges(edge_indices, num_nodes, num_neg_samples):
    """Generate negative edges for training."""
    neg_edge_index = torch.randint(0, num_nodes, (2, num_neg_samples))
    return neg_edge_index

def train(model, data, optimizer, num_epochs=100):
    """Train the link prediction model."""
    model.train()
    
    # Get all edge indices
    edge_indices = [data['node', f'relation_{i}', 'node'].edge_index for i in range(len(data.edge_types))]
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Generate negative edges
        neg_edge_index = generate_negative_edges(
            edge_indices, 
            data['node'].x.size(0), 
            edge_indices[0].size(1)
        )
        
        # Combine positive and negative edges
        edge_label_index = torch.cat([edge_indices[0], neg_edge_index], dim=1)
        edge_label = torch.cat([
            torch.ones(edge_indices[0].size(1)),
            torch.zeros(neg_edge_index.size(1))
        ])
        
        # Forward pass
        out = model(data['node'].x, edge_indices, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

def evaluate(model, data):
    """Evaluate the model on test edges."""
    model.eval()
    
    # Get all edge indices
    edge_indices = [data['node', f'relation_{i}', 'node'].edge_index for i in range(len(data.edge_types))]
    
    # Generate negative edges for testing
    neg_edge_index = generate_negative_edges(
        edge_indices, 
        data['node'].x.size(0), 
        edge_indices[0].size(1)
    )
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([edge_indices[0], neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(edge_indices[0].size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])
    
    with torch.no_grad():
        out = model(data['node'].x, edge_indices, edge_label_index)
        pred = torch.sigmoid(out)
        
        # Calculate metrics
        auc = roc_auc_score(edge_label.numpy(), pred.numpy())
        ap = average_precision_score(edge_label.numpy(), pred.numpy())
        
        return auc, ap

def main():
    # Load the graphs
    print("Loading graphs...")
    gexf_paths = [
        'res/bert_1_std_dev/publication_similarity_graph.gexf',
        'res/word2vec_1_std_dev/publication_similarity_graph.gexf'
    ]
    data = load_graphs(gexf_paths)
    
    # Split edges into train and test sets
    edge_index = data['node', 'relation_0', 'node'].edge_index.numpy().T
    train_edges, test_edges = train_test_split(edge_index, test_size=0.2, random_state=42)
    data['node', 'relation_0', 'node'].edge_index = torch.tensor(train_edges).t().contiguous()
    
    # Initialize model
    model = LinkPredictionModel(
        in_channels=len(gexf_paths),  # Number of relations
        hidden_channels=32,
        out_channels=16,
        num_relations=len(gexf_paths)
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train model
    print("Training model...")
    train(model, data, optimizer)
    
    # Evaluate model
    print("\nEvaluating model...")
    auc, ap = evaluate(model, data)
    print(f"Test AUC: {auc:.4f}")
    print(f"Test AP: {ap:.4f}")

if __name__ == "__main__":
    main() 