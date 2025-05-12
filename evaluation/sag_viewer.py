import pickle
import networkx as nx
import matplotlib.pyplot as plt

# # Load the graph
# with open('./brexit-test/polarization/sag.pckl', 'rb') as f:
#     G = pickle.load(f)

# with open("./brexit-test/polarization/int_to_node.pckl", "rb") as f:
#     int_to_node = pickle.load(f)

# # Load the graph
with open('../brexit-normalized-test/polarization/sag.pckl', 'rb') as f:
    G = pickle.load(f)

with open("../brexit-normalized-test/polarization/int_to_node.pckl", "rb") as f:
    int_to_node = pickle.load(f)

# === Build relabeled graph ===
G_named = nx.relabel_nodes(G, {
    node_id: int_to_node[node_id].replace("http://dbpedia.org/resource/", "").replace("_", " ").strip()
    for node_id in G.nodes
})

# === Define edge colors by polarity ===
edge_colors = ['red' if G_named[u][v]['weight'] < 0 else 'green' for u, v in G_named.edges()]

# === Draw the graph ===
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G_named, seed=42)

nx.draw(G_named, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue', font_size=9)

plt.title("Signed Attitude Graph (SAG) - POLAR")
plt.axis('off')

# Save the graph to a file
plt.savefig('gpt_sag_graph_output.png', format='png', dpi=300)
