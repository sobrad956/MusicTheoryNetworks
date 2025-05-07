import json

def count_nodes(json_file):
    """Count the number of nodes in a JSON network file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Check if the file has a 'nodes' field
            if 'nodes' in data:
                node_count = len(data['nodes'])
                print(f"Number of nodes in {json_file}: {node_count}")
            else:
                print(f"Error: {json_file} does not contain a 'nodes' field")
                
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    citation_file = "citation_net.json"
    count_nodes(citation_file) 