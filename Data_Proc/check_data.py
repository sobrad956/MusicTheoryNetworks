import networkx as nx
import json

def print_gexf_data():
    print("\nGEXF Data:")
    G = nx.read_gexf('Graphs/publication_similarity_graph_bert.gexf')
    for i, node in enumerate(list(G.nodes())[:5]):
        print(f"\nNode {node}:")
        print("Title:", repr(G.nodes[node].get('title', '')))
        print("Authors:", repr(G.nodes[node].get('authors', '')))
        print("Year:", repr(G.nodes[node].get('year', '')))

def print_library_data():
    print("\nLibrary Data:")
    with open('mto/all_library.json', 'r') as f:
        data = json.load(f)
        for i, (key, entry) in enumerate(list(data['publications'].items())[:5]):
            print(f"\nKey: {key}")
            print("Title:", repr(entry.get('title', '')))
            print("Authors:", repr(entry.get('authors', [])))
            print("Year:", repr(entry.get('year', '')))

print_gexf_data()
print_library_data() 