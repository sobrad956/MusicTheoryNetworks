import networkx as nx
import json

def load_mto_library():
    """Load the MTO library to get key mappings."""
    with open('mto/mto_library.json', 'r') as f:
        return json.load(f)

def create_key_mapping(mto_library):
    """Create a mapping from DOIs and titles to their keys in the publications dictionary."""
    key_mapping = {}
    for key, pub in mto_library['publications'].items():
        # Try to get DOI first
        doi = pub.get('doi')
        if doi:
            key_mapping[doi] = key
        # If no DOI, use title
        else:
            title = pub.get('title')
            if title:
                key_mapping[title] = key
    return key_mapping

def simplify_gexf(input_path, output_path, key_mapping):
    """Create a simplified version of the GEXF file with only the key attribute."""
    # Load the original graph
    G = nx.read_gexf(input_path)
    
    # Create a new graph
    G_simple = nx.Graph()
    
    # Add nodes with only the key attribute
    for node in G.nodes():
        # Get the original attributes
        attrs = G.nodes[node]
        
        # Try to find the key using DOI or title
        key = None
        if 'id' in attrs and attrs['id'] in key_mapping:
            key = key_mapping[attrs['id']]
        elif 'title' in attrs and attrs['title'] in key_mapping:
            key = key_mapping[attrs['title']]
        
        # Add node with only the key attribute if found
        if key:
            G_simple.add_node(node, key=key)
    
    # Add all edges with their original attributes
    for u, v, data in G.edges(data=True):
        if u in G_simple and v in G_simple:  # Only add edge if both nodes exist in new graph
            G_simple.add_edge(u, v, **data)
    
    # Save the simplified graph
    nx.write_gexf(G_simple, output_path)

def main():
    # Load MTO library and create key mapping
    print("Loading MTO library...")
    mto_library = load_mto_library()
    key_mapping = create_key_mapping(mto_library)
    
    # Process the GEXF file
    input_path = 'res/bert_1_std_dev/publication_similarity_graph.gexf'
    output_path = 'res/bert_1_std_dev/publication_similarity_graph_simple.gexf'
    
    print("Creating simplified GEXF file...")
    simplify_gexf(input_path, output_path, key_mapping)
    
    print(f"Simplified GEXF file saved to {output_path}")

if __name__ == "__main__":
    main() 