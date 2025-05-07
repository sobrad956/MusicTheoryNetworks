import json
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import gzip
import os
import csv

def load_raw_data():
    """Load the raw data to get node information."""
    with open('mto/all_library.json', 'r') as f:
        return json.load(f)

def load_citation_net(file_path):
    """Load citation network from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Check if the data is a dictionary with lists as values
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return data
        # If not, try to parse as nodes/edges format
        elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            # Convert to adjacency list format
            adj_list = {}
            for edge in data['edges']:
                source = edge['source']
                target = edge['target']
                if source not in adj_list:
                    adj_list[source] = []
                adj_list[source].append(target)
            return adj_list
        else:
            raise ValueError(f"Unexpected format in {file_path}")

def load_similarity_graph(json_path):
    """Load similarity graph from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Create a mapping from node IDs to indices
        node_to_idx = {}
        idx_to_node = {}
        for idx, node in enumerate(data['nodes']):
            node_id = node['id']
            node_to_idx[node_id] = idx
            idx_to_node[idx] = node_id
        
        # Create edge list with weights
        edges = []
        weights = []
        for edge in data['edges']:
            source = edge['source']
            target = edge['target']
            weight = edge.get('weight', 1.0)  # Default weight of 1.0 if not present
            edges.append((source, target))
            weights.append(weight)
        
        return edges, weights, node_to_idx, idx_to_node

def load_similarity_csv(csv_path):
    """Load similarity graph from gzipped CSV file."""
    edges = []
    with gzip.open(csv_path, 'rt') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:  # Make sure we have at least source and target
                source = row[0].strip()
                target = row[1].strip()
                edges.append((source, target))
    return edges

def print_graph_stats(graph, name):
    """Print statistics about a graph."""
    if isinstance(graph, dict):
        # For citation networks
        nodes = set(graph.keys())
        for targets in graph.values():
            nodes.update(targets)
        edges = sum(len(targets) for targets in graph.values())
        print(f"\n{name} statistics:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of edges: {edges}")
        return nodes
    elif isinstance(graph, tuple) and len(graph) == 3:
        # For JSON graphs with node mapping
        edges, node_to_idx, idx_to_node = graph
        print(f"\n{name} statistics:")
        print(f"Number of nodes: {len(node_to_idx)}")
        print(f"Number of edges: {len(edges)}")
        return set(node_to_idx.keys())
    elif isinstance(graph, list):
        # For CSV edges
        nodes = set()
        for source, target in graph:
            nodes.add(source)
            nodes.add(target)
        print(f"\n{name} statistics:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of edges: {len(graph)}")
        return nodes
    else:
        raise ValueError(f"Unexpected graph type: {type(graph)}")

def create_node_mapping():
    """Create a mapping from paper IDs to node indices."""
    node_to_idx = {}
    idx_to_node = {}
    
    # First, load the BERT graph to get the set of valid nodes
    bert_nodes = set()
    try:
        bert_edges, bert_weights, bert_node_to_idx, _ = load_similarity_graph('Graphs/reformatted/publication_similarity_graph_bert.json')
        bert_nodes = set(bert_node_to_idx.keys())
        print_graph_stats((bert_edges, bert_node_to_idx, {}), "BERT similarity graph")
    except Exception as e:
        print(f"Warning: Error loading publication_similarity_graph_bert.json: {e}")
    
    # Create a set of all unique node IDs from all graphs
    all_nodes = set()
    
    # Add nodes from citation networks (only if they exist in BERT graph)
    try:
        citation_net = load_citation_net('Graphs/citation_net.json')
        citation_nodes = set(citation_net.keys())
        for targets in citation_net.values():
            citation_nodes.update(targets)
        # Filter nodes to only those in BERT graph
        citation_nodes = citation_nodes.intersection(bert_nodes)
        all_nodes.update(citation_nodes)
        print(f"\nOne hop citation network statistics (filtered):")
        print(f"Number of nodes: {len(citation_nodes)}")
        print(f"Number of edges: {sum(1 for source, targets in citation_net.items() if source in citation_nodes and any(t in citation_nodes for t in targets))}")
    except Exception as e:
        print(f"Warning: Error loading citation_net.json: {e}")
    
    try:
        citation_net_no_hop = load_citation_net('Graphs/citation_net no-hop.json')
        citation_no_hop_nodes = set(citation_net_no_hop.keys())
        for targets in citation_net_no_hop.values():
            citation_no_hop_nodes.update(targets)
        # Filter nodes to only those in BERT graph
        citation_no_hop_nodes = citation_no_hop_nodes.intersection(bert_nodes)
        all_nodes.update(citation_no_hop_nodes)
        print(f"\nDirect citation network statistics (filtered):")
        print(f"Number of nodes: {len(citation_no_hop_nodes)}")
        print(f"Number of edges: {sum(1 for source, targets in citation_net_no_hop.items() if source in citation_no_hop_nodes and any(t in citation_no_hop_nodes for t in targets))}")
    except Exception as e:
        print(f"Warning: Error loading citation_net no-hop.json: {e}")
    
    # Add nodes from similarity graphs
    try:
        w2v_edges, w2v_weights, w2v_node_to_idx, _ = load_similarity_graph('Graphs/reformatted/publication_similarity_graph_w2v.json')
        w2v_nodes = set(w2v_node_to_idx.keys())
        # Filter nodes to only those in BERT graph
        w2v_nodes = w2v_nodes.intersection(bert_nodes)
        all_nodes.update(w2v_nodes)
        print(f"\nWord2Vec similarity graph statistics (filtered):")
        print(f"Number of nodes: {len(w2v_nodes)}")
        print(f"Number of edges: {sum(1 for u, v in w2v_edges if u in w2v_nodes and v in w2v_nodes)}")
    except Exception as e:
        print(f"Warning: Error loading publication_similarity_graph_w2v.json: {e}")
    
    # Add nodes from CSV file (only if they exist in BERT graph)
    try:
        similarity_edges = load_similarity_csv('Graphs/paperGraph.postThresh.mean.csv.gz')
        csv_nodes = set()
        for source, target in similarity_edges:
            csv_nodes.add(source)
            csv_nodes.add(target)
        # Filter nodes to only those in BERT graph
        csv_nodes = csv_nodes.intersection(bert_nodes)
        all_nodes.update(csv_nodes)
        print(f"\nKeywords similarity graph statistics (filtered):")
        print(f"Number of nodes: {len(csv_nodes)}")
        print(f"Number of edges: {sum(1 for u, v in similarity_edges if u in csv_nodes and v in csv_nodes)}")
    except Exception as e:
        print(f"Warning: Error loading paperGraph.postThresh.mean.csv.gz: {e}")
    
    # Create mapping for all unique nodes
    for idx, node in enumerate(sorted(all_nodes)):
        if node:  # Only include non-empty nodes
            node_to_idx[node] = idx
            idx_to_node[idx] = node
    
    return node_to_idx, idx_to_node

def make_undirected(edges):
    """Convert a directed edge list to undirected by adding reciprocal edges."""
    edge_set = set((source, target) for source, target in edges)
    edge_set.update((target, source) for source, target in edges)
    return list(edge_set)

def add_edges_to_multiplex(data, G, node_mapping, edge_type):
    edge_index = []
    edge_weight = []
    for u, v, edata in G.edges(data=True):
        if u in node_mapping and v in node_mapping:
            edge_index.append([node_mapping[u], node_mapping[v]])
            # Always store as 'weight' in the multiplex
            if 'weight' in edata:
                edge_weight.append(edata['weight'])
            elif 'val' in edata:
                edge_weight.append(edata['val'])
            else:
                edge_weight.append(1.0)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).view(-1, 1)
        data[edge_type].edge_index = edge_index
        data[edge_type].edge_attr = edge_weight
        data[edge_type].weight = edge_weight
    return data

def check_edges_and_weights(graphs, multiplex_data, node_mapping, edge_type_mapping):
    total_missing = 0
    total_mismatched = 0
    for graph_name, G in graphs.items():
        edge_type = ('node', edge_type_mapping[graph_name], 'node')
        if edge_type not in multiplex_data.edge_index_dict:
            print(f"[WARNING] Edge type {edge_type} not found in multiplex network.")
            continue
        edge_index = multiplex_data[edge_type].edge_index
        edge_weight = multiplex_data[edge_type].edge_weight.squeeze().tolist()
        edge_lookup = {(edge_index[0, i].item(), edge_index[1, i].item()): edge_weight[i] for i in range(edge_index.shape[1])}
        missing = 0
        mismatched = 0
        for u, v, edata in G.edges(data=True):
            u_idx = node_mapping.get(u)
            v_idx = node_mapping.get(v)
            if u_idx is None or v_idx is None:
                missing += 1
                continue
            key = (u_idx, v_idx)
            key_rev = (v_idx, u_idx)
            orig_w = edata.get('weight', edata.get('val', 1.0))
            found = False
            for k in [key, key_rev]:
                if k in edge_lookup:
                    found = True
                    if not (abs(edge_lookup[k] - orig_w) < 1e-8):
                        mismatched += 1
                    break
            if not found:
                missing += 1
        print(f"Graph: {graph_name}")
        print(f"  Edges in original graph: {G.number_of_edges()}")
        print(f"  Edges in multiplex network (edge type): {edge_index.shape[1]}")
        if missing == 0:
            print(f"  [OK] All edges from {graph_name} are present in the multiplex network.")
        else:
            print(f"  [MISSING] {missing} edges from {graph_name} are missing in the multiplex network.")
        if mismatched == 0:
            print(f"  [OK] All edge weights match for {graph_name}.")
        else:
            print(f"  [MISMATCH] {mismatched} edge weights do not match for {graph_name}.")
        total_missing += missing
        total_mismatched += mismatched
    print(f"Total missing edges: {total_missing}")
    print(f"Total mismatched edge weights: {total_mismatched}")

def print_total_edges(multiplex_data):
    total_edges = 0
    for edge_type in multiplex_data.edge_index_dict:
        total_edges += multiplex_data[edge_type].edge_index.shape[1]
    print(f"Total number of edges of any type in the multiplex network: {total_edges}")

def create_multiplex_network():
    # Create node mapping
    node_to_idx, idx_to_node = create_node_mapping()
    num_nodes = len(node_to_idx)
    
    print(f"\nCreated node mapping with {num_nodes} nodes")
    
    # Initialize heterogeneous graph
    data = HeteroData()
    
    # Add nodes
    data['node'].x = torch.zeros((num_nodes, 1))  # Using degree as feature
    
    # Load and process each graph
    # 1. One hop citation network
    try:
        citation_net = load_citation_net('Graphs/citation_net.json')
        citation_edges = []
        for source, targets in citation_net.items():
            if source in node_to_idx:
                source_idx = node_to_idx[source]
                if isinstance(targets, list):  # Make sure targets is a list
                    for target in targets:
                        if target in node_to_idx:
                            target_idx = node_to_idx[target]
                            citation_edges.append((source_idx, target_idx))
        
        if citation_edges:
            src, dst = zip(*citation_edges)
            data['node', 'one_hop_citation', 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            # Add uniform weights for citation edges
            data['node', 'one_hop_citation', 'node'].edge_weight = torch.ones(len(citation_edges), dtype=torch.float)
    except Exception as e:
        print(f"Warning: Error processing citation_net.json: {e}")
        citation_edges = []
    
    # 2. Direct citation network
    try:
        citation_net_no_hop = load_citation_net('Graphs/citation_net no-hop.json')
        citation_no_hop_edges = []
        for source, targets in citation_net_no_hop.items():
            if source in node_to_idx:
                source_idx = node_to_idx[source]
                if isinstance(targets, list):  # Make sure targets is a list
                    for target in targets:
                        if target in node_to_idx:
                            target_idx = node_to_idx[target]
                            citation_no_hop_edges.append((source_idx, target_idx))
        
        if citation_no_hop_edges:
            src, dst = zip(*citation_no_hop_edges)
            data['node', 'direct_citation', 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            # Add uniform weights for citation edges
            data['node', 'direct_citation', 'node'].edge_weight = torch.ones(len(citation_no_hop_edges), dtype=torch.float)
    except Exception as e:
        print(f"Warning: Error processing citation_net no-hop.json: {e}")
        citation_no_hop_edges = []
    
    # 3. BERT similarity graph
    try:
        bert_edges, bert_weights, bert_node_to_idx, _ = load_similarity_graph('Graphs/reformatted/publication_similarity_graph_bert.json')
        # Add both (u, v) and (v, u) for each edge
        bert_edges_idx = []
        bert_weights_idx = []
        for (source, target), weight in zip(bert_edges, bert_weights):
            if source in node_to_idx and target in node_to_idx:
                source_idx = node_to_idx[source]
                target_idx = node_to_idx[target]
                # Add (u, v)
                bert_edges_idx.append((source_idx, target_idx))
                bert_weights_idx.append(weight)
                # Add (v, u) if not self-loop
                if source_idx != target_idx:
                    bert_edges_idx.append((target_idx, source_idx))
                    bert_weights_idx.append(weight)
        if bert_edges_idx:
            src, dst = zip(*bert_edges_idx)
            data['node', 'bert', 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['node', 'bert', 'node'].edge_weight = torch.tensor(bert_weights_idx, dtype=torch.float)
    except Exception as e:
        print(f"Warning: Error processing publication_similarity_graph_bert.json: {e}")
        bert_edges_idx = []
    
    # 4. Word2Vec similarity graph
    try:
        w2v_edges, w2v_weights, w2v_node_to_idx, _ = load_similarity_graph('Graphs/reformatted/publication_similarity_graph_w2v.json')
        # Add both (u, v) and (v, u) for each edge
        w2v_edges_idx = []
        w2v_weights_idx = []
        for (source, target), weight in zip(w2v_edges, w2v_weights):
            if source in node_to_idx and target in node_to_idx:
                source_idx = node_to_idx[source]
                target_idx = node_to_idx[target]
                # Add (u, v)
                w2v_edges_idx.append((source_idx, target_idx))
                w2v_weights_idx.append(weight)
                # Add (v, u) if not self-loop
                if source_idx != target_idx:
                    w2v_edges_idx.append((target_idx, source_idx))
                    w2v_weights_idx.append(weight)
        if w2v_edges_idx:
            src, dst = zip(*w2v_edges_idx)
            data['node', 'w2v', 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['node', 'w2v', 'node'].edge_weight = torch.tensor(w2v_weights_idx, dtype=torch.float)
    except Exception as e:
        print(f"Warning: Error processing publication_similarity_graph_w2v.json: {e}")
        w2v_edges_idx = []
    
    # 5. Keywords similarity graph
    try:
        df = pd.read_csv('Graphs/paperGraph.postThresh.mean.csv.gz', compression='gzip')
        keyword_edges_idx = []
        keyword_weights_idx = []
        for _, row in df.iterrows():
            source = row['Source']
            target = row['Target']
            weight = row['val']
            if source in node_to_idx and target in node_to_idx:
                source_idx = node_to_idx[source]
                target_idx = node_to_idx[target]
                # Add (u, v)
                keyword_edges_idx.append((source_idx, target_idx))
                keyword_weights_idx.append(weight)
                # Add (v, u) if not self-loop
                if source_idx != target_idx:
                    keyword_edges_idx.append((target_idx, source_idx))
                    keyword_weights_idx.append(weight)
        if keyword_edges_idx:
            src, dst = zip(*keyword_edges_idx)
            data['node', 'keywords', 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['node', 'keywords', 'node'].edge_weight = torch.tensor(keyword_weights_idx, dtype=torch.float)
    except Exception as e:
        print(f"Warning: Error processing paperGraph.postThresh.mean.csv.gz: {e}")
        keyword_edges_idx = []
    
    # Print final statistics
    print("\nMultiplex network statistics:")
    print(f"Total nodes: {num_nodes}")
    print(f"One hop citation edges: {len(citation_edges)}")
    print(f"Direct citation edges: {len(citation_no_hop_edges)}")
    print(f"BERT similarity edges (undirected): {len(bert_edges_idx)}")
    print(f"Word2Vec similarity edges (undirected): {len(w2v_edges_idx)}")
    print(f"Keywords similarity edges (undirected): {len(keyword_edges_idx)}")
    
    # Load or construct the graphs dictionary and edge_type_mapping
    import networkx as nx
    graphs = {}
    # Load BERT similarity graph
    G_bert = nx.read_gexf('Graphs/publication_similarity_graph_bert.gexf')
    graphs['publication_similarity_graph_bert'] = G_bert
    # Load one hop citation network
    with open('Graphs/citation_net.json', 'r') as f:
        citation_data = json.load(f)
        G_citation = nx.node_link_graph(citation_data, edges="edges")
    graphs['citation_net'] = G_citation
    # Load direct citation network
    with open('Graphs/citation_net no-hop.json', 'r') as f:
        citation_nohop_data = json.load(f)
        G_citation_no_hop = nx.node_link_graph(citation_nohop_data, edges="edges")
    graphs['citation_net no-hop'] = G_citation_no_hop
    # Load Word2Vec similarity graph
    G_w2v = nx.read_gexf('Graphs/publication_similarity_graph_w2v.gexf')
    graphs['publication_similarity_graph_w2v'] = G_w2v
    # Load keywords similarity graph
    df = pd.read_csv('Graphs/paperGraph.postThresh.mean.csv.gz', compression='gzip')
    G_keyword = nx.from_pandas_edgelist(df, 'Source', 'Target', 'val')
    graphs['keyword_similarity'] = G_keyword
    # Load normalized graph if needed
    try:
        G_norm = nx.read_gexf('Graphs/normalized_graph.gexf')
        graphs['normalized_graph'] = G_norm
    except Exception:
        pass
    edge_type_mapping = {
        'citation_net no-hop': 'direct_citation',
        'citation_net': 'one_hop_citation',
        'publication_similarity_graph_bert': 'bert',
        'publication_similarity_graph_w2v': 'w2v',
        'normalized_graph': 'normalized',
        'keyword_similarity': 'keywords'
    }
    
    print("Edge types present in multiplex network:")
    print(list(data.edge_index_dict.keys()))
    # Update edge_type_mapping here if needed to match these keys
    # Then run the check as before
    print_total_edges(data)
    check_edges_and_weights(graphs, data, node_to_idx, edge_type_mapping)

    # Find a pair of nodes with all 5 edge types
    edge_types = ['one_hop_citation', 'direct_citation', 'bert', 'w2v', 'keywords']
    edge_type_keys = [('node', et, 'node') for et in edge_types]
    found_pair = None
    for u in range(len(node_to_idx)):
        for v in range(len(node_to_idx)):
            if u == v:
                continue
            has_all = True
            weights = {}
            for et_key, et_name in zip(edge_type_keys, edge_types):
                if et_key in data.edge_index_dict:
                    edge_index = data[et_key].edge_index
                    edge_weight = data[et_key].edge_weight
                    # Check for (u, v) or (v, u)
                    mask = ((edge_index[0] == u) & (edge_index[1] == v)) | ((edge_index[0] == v) & (edge_index[1] == u))
                    if mask.any():
                        idx = mask.nonzero(as_tuple=True)[0][0].item()
                        weights[et_name] = edge_weight[idx].item()
                    else:
                        has_all = False
                        break
                else:
                    has_all = False
                    break
            if has_all:
                found_pair = (u, v, weights)
                break
        if found_pair:
            break
    if found_pair:
        u, v, weights = found_pair
        print(f"\nPair of nodes with all 5 edge types: {u} <-> {v}")
        print(f"Original IDs: {idx_to_node[u]} <-> {idx_to_node[v]}")
        for et in edge_types:
            print(f"  Edge type '{et}': weight = {weights[et]}")
    else:
        print("No pair of nodes found with all 5 edge types.")

    return data, node_to_idx, idx_to_node, graphs, edge_type_mapping

if __name__ == '__main__':
    data, node_to_idx, idx_to_node, graphs, edge_type_mapping = create_multiplex_network()
    # Save the multiplex network
    torch.save(data, 'Graphs/multiplex_network.pt')
    # Save the node mappings
    with open('Graphs/node_mappings.json', 'w') as f:
        json.dump({
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node
        }, f, indent=4)
    print_total_edges(data)
    check_edges_and_weights(graphs, data, node_to_idx, edge_type_mapping) 