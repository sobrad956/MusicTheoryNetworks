import networkx as nx

def print_node_info(file_path):
    print(f"\nChecking node info in {file_path}")
    G = nx.read_gexf(file_path)
    print('First 5 nodes and their attributes:')
    for i, node in enumerate(list(G.nodes())[:5]):
        print(f'\nNode {node}:')
        print('Direct attributes:', G.nodes[node])
        if 'attvalues' in G.nodes[node]:
            print('Attribute values:')
            for attvalue in G.nodes[node]['attvalues']:
                print(attvalue)

print_node_info('Graphs/publication_similarity_graph_bert.gexf')
print_node_info('Graphs/publication_similarity_graph_w2v.gexf') 