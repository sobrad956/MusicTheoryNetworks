import networkx as nx
import json
import os

def load_graph_and_publications(graph_path, publications_dir):
    """Load a graph and its corresponding publications."""
    # Load the graph
    G = nx.read_gexf(graph_path)
    
    # Load all publications
    publications = []
    for filename in os.listdir(publications_dir):
        if filename.endswith('.json'):
            with open(os.path.join(publications_dir, filename), 'r') as f:
                data = json.load(f)
                publications.append({
                    'title': data.get('title', ''),
                    'abstract': data.get('abstract', ''),
                    'source': os.path.join(publications_dir, filename)
                })
    
    return G, publications

def find_different_edges(word2vec_graph, bert_graph, publications):
    """Find edges with significantly different weights between the two graphs."""
    # Get all edges that exist in both graphs
    common_edges = set(word2vec_graph.edges()) & set(bert_graph.edges())
    
    # Calculate weight differences
    weight_differences = []
    for u, v in common_edges:
        w2v_weight = word2vec_graph[u][v]['weight']
        bert_weight = bert_graph[u][v]['weight']
        weight_diff = bert_weight - w2v_weight  # Positive means BERT > Word2Vec, negative means BERT < Word2Vec
        weight_differences.append((u, v, w2v_weight, bert_weight, weight_diff))
    
    # Sort by absolute weight difference
    weight_differences.sort(key=lambda x: abs(x[4]), reverse=True)
    
    return weight_differences

def main():
    # Load both graphs and publications
    word2vec_graph, word2vec_pubs = load_graph_and_publications(
        'res/word2vec_1_std_dev/publication_similarity_graph.gexf',
        'mto/data'
    )
    bert_graph, bert_pubs = load_graph_and_publications(
        'res/bert_1_std_dev/publication_similarity_graph.gexf',
        'mto/data'
    )
    
    # Create a mapping from node IDs to publication indices
    node_to_pub = {}
    for node in word2vec_graph.nodes():
        # Get the title from the node attributes
        title = word2vec_graph.nodes[node].get('title', '')
        # Find the publication with this title
        for i, pub in enumerate(word2vec_pubs):
            if pub['title'] == title:
                node_to_pub[node] = i
                break
    
    # Find edges with different weights
    weight_differences = find_different_edges(word2vec_graph, bert_graph, word2vec_pubs)
    
    # Print the top 5 edges where BERT score is higher than Word2Vec
    print("\nTop 5 edges where BERT similarity is higher than Word2Vec:")
    print("=" * 80)
    bert_higher = [diff for diff in weight_differences if diff[4] > 0][:5]
    bert_lower = weight_differences[:-5]
    for u, v, w2v_weight, bert_weight, diff in bert_higher:
        print(f"\nEdge between nodes {u} and {v}:")
        print(f"Word2Vec similarity: {w2v_weight:.4f}")
        print(f"BERT similarity: {bert_weight:.4f}")
        print(f"Difference (BERT - Word2Vec): {diff:.4f}")
        
        # Get the publication indices
        pub1_idx = node_to_pub.get(u)
        pub2_idx = node_to_pub.get(v)
        
        if pub1_idx is not None and pub2_idx is not None:
            # Print paper information
            print("\nPaper 1:")
            print(f"Title: {word2vec_pubs[pub1_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub1_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub1_idx]['source']}")
            
            print("\nPaper 2:")
            print(f"Title: {word2vec_pubs[pub2_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub2_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub2_idx]['source']}")
        else:
            print("\nWarning: Could not find publication information for one or both nodes")
        print("=" * 80)

    for u, v, w2v_weight, bert_weight, diff in bert_lower:
        print(f"\nEdge between nodes {u} and {v}:")
        print(f"Word2Vec similarity: {w2v_weight:.4f}")
        print(f"BERT similarity: {bert_weight:.4f}")
        print(f"Difference (BERT - Word2Vec): {diff:.4f}")
        
        # Get the publication indices
        pub1_idx = node_to_pub.get(u)
        pub2_idx = node_to_pub.get(v)
        
        if pub1_idx is not None and pub2_idx is not None:
            # Print paper information
            print("\nPaper 1:")
            print(f"Title: {word2vec_pubs[pub1_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub1_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub1_idx]['source']}")
            
            print("\nPaper 2:")
            print(f"Title: {word2vec_pubs[pub2_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub2_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub2_idx]['source']}")
        else:
            print("\nWarning: Could not find publication information for one or both nodes")
        print("=" * 80)
    
    # Print the top 5 edges where Word2Vec score is higher than BERT
    print("\nTop 5 edges where Word2Vec similarity is higher than BERT:")
    print("=" * 80)
    w2v_higher = [diff for diff in weight_differences if diff[4] < 0][:5]
    for u, v, w2v_weight, bert_weight, diff in w2v_higher:
        print(f"\nEdge between nodes {u} and {v}:")
        print(f"Word2Vec similarity: {w2v_weight:.4f}")
        print(f"BERT similarity: {bert_weight:.4f}")
        print(f"Difference (BERT - Word2Vec): {diff:.4f}")
        
        # Get the publication indices
        pub1_idx = node_to_pub.get(u)
        pub2_idx = node_to_pub.get(v)
        
        if pub1_idx is not None and pub2_idx is not None:
            # Print paper information
            print("\nPaper 1:")
            print(f"Title: {word2vec_pubs[pub1_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub1_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub1_idx]['source']}")
            
            print("\nPaper 2:")
            print(f"Title: {word2vec_pubs[pub2_idx]['title']}")
            print(f"Abstract: {word2vec_pubs[pub2_idx]['abstract']}")
            print(f"Source: {word2vec_pubs[pub2_idx]['source']}")
        else:
            print("\nWarning: Could not find publication information for one or both nodes")
        print("=" * 80)

if __name__ == "__main__":
    main() 