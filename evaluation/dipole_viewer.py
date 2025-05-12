import pickle
import networkx as nx
import matplotlib.pyplot as plt

# # Load the data
# with open("./brexit-test/polarization/attitudes.pckl", "rb") as f:
#     attitude_data = pickle.load(f)

# with open("./brexit-test/polarization/int_to_node.pckl", "rb") as f:
#     int_to_node = pickle.load(f)

# # Load the data
# with open("../brexit-normalized-test/polarization/attitudes.pckl", "rb") as f:
#     attitude_data = pickle.load(f)

# with open("../brexit-normalized-test/polarization/int_to_node.pckl", "rb") as f:
#     int_to_node = pickle.load(f)

# Load the data
with open("../test-brexit/gpt_polarization/attitudes.pckl", "rb") as f:
    attitude_data = pickle.load(f)

with open("../test-brexit/gpt_polarization/int_to_node.pckl", "rb") as f:
    int_to_node = pickle.load(f)


# Step 2: Construct topic graph from attitudes
def build_topic_graph(attitudes, int_to_node):
    G = nx.DiGraph()
    for entry in attitudes:
        source, target = entry['dipole']
        topic = entry['topic']['nps'][0] if entry['topic']['nps'] else "Unknown"
        pi = entry.get("pi_res", entry.get("pi"))

       # Get attitude values (default to 0 if missing)
        att_i = entry.get('atts_fi', [0])[0] if entry.get('atts_fi') else 0
        att_j = entry.get('atts_fj', [0])[0] if entry.get('atts_fj') else 0
        
        # Define attitude category
        if pi > 0.6:
            label = "Polarized"
            color = "red"
        elif pi <= 0:
            # Check if both attitudes are close
            if abs(att_i - att_j) > 0.5:
                label = "Neutral Agreement"
                color = "blue"  # Using blue for neutral agreement
            elif att_i > 0 and att_j > 0:
                #print(f"atts_fi: {att_i}, atts_fj: {att_j}")
                label = "Agreement Positive"
                color = "green"
            else:
                label = "Agreement Negative"
                color = "orange"
        else:
            label = "Mixed"
            color = "gray"
        
        # Add edge
        G.add_edge(
            int_to_node[source].replace("http://dbpedia.org/resource/", ""),
            int_to_node[target].replace("http://dbpedia.org/resource/", ""),
            topic=topic,
            pi=pi,
            label=label,
            color=color
        )
    return G

G_topic = build_topic_graph(attitude_data, int_to_node)


# Step 3: Visualize a small subgraph
def draw_subgraph(G, max_edges=500, output_file="un_dipole_graph.png"):
    plt.figure(figsize=(14, 10))
    subgraph = G.edge_subgraph(list(G.edges())[:max_edges]).copy()
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)  # Increase `k` for more spread
    
    # Draw edges with colors/widths
    edge_colors = [data['color'] for _, _, data in subgraph.edges(data=True)]
    nx.draw(subgraph, pos, with_labels=True, edge_color=edge_colors, 
            node_size=700, font_size=10, width=2)
    
    # Add labels with offset (0.5 = middle, 0.8 = closer to target)
    edge_labels = {(u, v): data['topic'] for u, v, data in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        subgraph, pos, edge_labels=edge_labels,
        label_pos=0.5,  # Adjust to 0.6-0.8 to avoid overlaps
        font_size=11, 
    )
    plt.savefig(output_file, dpi=300)
    plt.close()

draw_subgraph(G_topic)


from collections import defaultdict

def summarize_by_topic(attitudes, int_to_node):
    topic_stats = defaultdict(list)

    for entry in attitudes:
        topic = entry['topic']['nps'][0] if entry['topic']['nps'] else "Unknown"
        pi = entry.get("pi_res", entry.get("pi"))
        topic_stats[topic].append(pi)

    print(f"{'Topic':<30} {'Avg π':>6} {'Count':>6}")
    print("=" * 45)
    for topic, pis in sorted(topic_stats.items(), key=lambda x: len(x[1]), reverse=True):
        avg_pi = sum(pis) / len(pis)
        print(f"{topic:<30} {avg_pi:>6.2f} {len(pis):>6}")

summarize_by_topic(attitude_data, int_to_node)