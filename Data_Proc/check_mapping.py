import networkx as nx
from FINAL_CODE.create_multiplex_network import create_node_mapping

def check_mappings():
    node_to_idx, idx_to_node, title_to_idx = create_node_mapping()
    
    # Load BERT graph and check a few titles
    G_bert = nx.read_gexf('Graphs/publication_similarity_graph_bert.gexf')
    print("\nChecking BERT graph titles:")
    for i, node in enumerate(list(G_bert.nodes())[:5]):
        title = G_bert.nodes[node].get('title', '')
        print(f"\nTitle: {title}")
        print(f"In title_to_idx: {title in title_to_idx}")
        if title in title_to_idx:
            print(f"Index: {title_to_idx[title]}")
            print(f"Node in node_to_idx: {title in node_to_idx}")
            if title in node_to_idx:
                print(f"Same index: {title_to_idx[title] == node_to_idx[title]}")

check_mappings() 