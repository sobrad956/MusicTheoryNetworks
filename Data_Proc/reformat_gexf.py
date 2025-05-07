import networkx as nx
import json
import os

def load_library_data():
    """Load the library data to get node information."""
    with open('mto/all_library.json', 'r') as f:
        data = json.load(f)
        return data['publications']

def normalize_title(title):
    """Normalize a title string."""
    # Remove HTML tags
    title = title.replace('<i>', '').replace('</i>', '')
    # Remove quotes
    title = title.replace('"', '').replace('"', '')
    # Normalize whitespace
    title = ' '.join(title.split())
    # Convert to lowercase
    return title.lower().strip()

def normalize_author(author):
    """Normalize an author string."""
    # Remove dots and commas
    author = author.replace('.', '').replace(',', '')
    # Handle middle names/initials
    parts = author.split()
    if len(parts) > 2:
        # Keep first and last name, remove middle parts
        author = f"{parts[0]} {parts[-1]}"
    # Normalize whitespace and convert to lowercase
    return ' '.join(author.split()).lower().strip()

def find_matching_key(library_data, title, authors, year, doi=None):
    """Find the matching key in library data based on title, authors, year, and doi."""
    # Manual mapping for the special case
    if title == "What Modular Analysis Can Tell Us About Musical Modeling in the Renaissance" and year == "2013":
        return "Schubert2013What_Modular_Analysis"
    
    # First try matching by DOI if available
    if doi and doi in library_data:
        return doi
    
    # Normalize the input
    norm_title = normalize_title(title)
    norm_authors = [normalize_author(a) for a in authors.split(',')]
    norm_year = str(year).strip()
    
    # Try to find exact matches first
    for key, entry in library_data.items():
        if 'title' in entry and 'authors' in entry and 'year' in entry:
            entry_title = normalize_title(entry['title'])
            entry_authors = [normalize_author(a) for a in entry['authors']]
            entry_year = str(entry['year']).strip()
            
            # Check for exact title match
            if entry_title == norm_title:
                # Check if any author matches
                if any(na in entry_authors or any(na in ea for ea in entry_authors) for na in norm_authors):
                    # Check year
                    if entry_year == norm_year:
                        return key
    
    # If no exact match, try partial matches
    for key, entry in library_data.items():
        if 'title' in entry and 'authors' in entry and 'year' in entry:
            entry_title = normalize_title(entry['title'])
            entry_authors = [normalize_author(a) for a in entry['authors']]
            entry_year = str(entry['year']).strip()
            
            # Check for significant title overlap
            title_words = set(norm_title.split())
            entry_words = set(entry_title.split())
            common_words = title_words & entry_words
            if len(common_words) >= min(len(title_words), len(entry_words)) * 0.8:  # 80% overlap
                # Check if any author matches
                if any(na in entry_authors or any(na in ea for ea in entry_authors) for na in norm_authors):
                    # Check year
                    if entry_year == norm_year:
                        return key
    
    return None

def reformat_gexf(input_path, output_path, library_data):
    """Reformat a GEXF file to use library keys as node labels and convert to JSON format."""
    # Load the original graph
    G = nx.read_gexf(input_path)
    
    # Create a new graph
    new_G = nx.Graph()
    
    # Process each node
    node_mapping = {}
    unmatched_nodes = []
    for node in G.nodes():
        title = G.nodes[node].get('title', '')
        authors = G.nodes[node].get('authors', '')
        year = G.nodes[node].get('year', '')
        doi = G.nodes[node].get('doi', '')
        
        if title and authors and year:
            # Find matching key in library data
            key = find_matching_key(library_data, title, authors, year, doi)
            if key:
                node_mapping[node] = key
                # Add node with library key
                new_G.add_node(key, mto_paper=True)
            else:
                unmatched_nodes.append((node, title, authors, year))
    
    # Process each edge
    for u, v, data in G.edges(data=True):
        if u in node_mapping and v in node_mapping:
            new_u = node_mapping[u]
            new_v = node_mapping[v]
            weight = data.get('weight', 1.0)  # Get weight, default to 1.0 if not present
            new_G.add_edge(new_u, new_v, weight=weight)
    
    # Convert to JSON format
    json_data = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": node, "mto_paper": True} for node in new_G.nodes()],
        "edges": [{"source": u, "target": v, "weight": data.get('weight', 1.0)} for u, v, data in new_G.edges(data=True)]
    }
    
    # Save as JSON
    output_json_path = output_path.replace('.gexf', '.json')
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    # Print statistics
    print(f"\nReformatted {input_path}:")
    print(f"Original nodes: {G.number_of_nodes()}")
    print(f"New nodes: {new_G.number_of_nodes()}")
    print(f"Original edges: {G.number_of_edges()}")
    print(f"New edges: {new_G.number_of_edges()}")
    print(f"Nodes matched to library keys: {len(node_mapping)}")
    
    # Print some example matches
    print("\nExample matches:")
    for i, (old_node, new_key) in enumerate(list(node_mapping.items())[:5]):
        print(f"\nOriginal node {old_node}:")
        print(f"Title: {G.nodes[old_node].get('title', '')}")
        print(f"Authors: {G.nodes[old_node].get('authors', '')}")
        print(f"Year: {G.nodes[old_node].get('year', '')}")
        print(f"DOI: {G.nodes[old_node].get('doi', '')}")
        print(f"Matched to key: {new_key}")
        print(f"Library data:")
        print(f"Title: {library_data[new_key].get('title', '')}")
        print(f"Authors: {library_data[new_key].get('authors', [])}")
        print(f"Year: {library_data[new_key].get('year', '')}")
    
    # Print some unmatched nodes
    if unmatched_nodes:
        print("\nSample of unmatched nodes:")
        for node, title, authors, year in unmatched_nodes[:5]:
            print(f"\nNode {node}:")
            print(f"Title: {title}")
            print(f"Authors: {authors}")
            print(f"Year: {year}")

def main():
    # Load library data
    library_data = load_library_data()
    
    # Create output directory if it doesn't exist
    os.makedirs('Graphs/reformatted', exist_ok=True)
    
    # Reformat BERT similarity graph
    reformat_gexf(
        'Graphs/publication_similarity_graph_bert.gexf',
        'Graphs/reformatted/publication_similarity_graph_bert.gexf',
        library_data
    )
    
    # Reformat Word2Vec similarity graph
    reformat_gexf(
        'Graphs/publication_similarity_graph_w2v.gexf',
        'Graphs/reformatted/publication_similarity_graph_w2v.gexf',
        library_data
    )

if __name__ == '__main__':
    main() 