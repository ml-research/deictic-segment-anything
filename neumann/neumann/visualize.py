import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE


def plot_atoms(x_atom, atoms, path):
    x_atom = x_atom.detach().cpu().numpy()
    labels = [str(atom) for atom in atoms]
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_atom)
    fig, ax = plt.subplots(figsize=(30,30))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
    for i, label in enumerate(labels):
        ax.annotate(label, (X_reduced[i,0], X_reduced[i,1]))
    plt.savefig(path)

def plot_infer_embeddings(x_atom_list, atoms):
    for i, x_atom in enumerate(x_atom_list):
        plot_atoms(x_atom, atoms, 'imgs/x_atom_' + str(i) + '.png')


def plot_proof_history(V_list, atoms, infer_step, dataset, mode='graph'):
    if dataset == 'graph':
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
    # extract first batch
    vs_img = np.round(np.array([vs[0] for vs in V_list]),2)
    print(vs_img)
    vmax = vs_img.max()
    im = ax.imshow(vs_img, cmap="Blues")
                ##m = ax.imshow(vs_img, cmap="plasma")
    ax.set_xticks(np.arange(len(atoms)))
    ax.set_yticks(np.arange(infer_step+1))
    plt.yticks(fontname = "monospace", fontsize=12)
    plt.xticks(fontname = "monospace", fontsize=11)
    ax.set_xticklabels([str(x) for x in atoms])
    ax.set_yticklabels(["v_{}".format(i) for i in range(infer_step+1)])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    plt.rcParams.update({'font.size': 10})
    for i in range(infer_step+1):
        for j in range(len(atoms)):
            if vs_img[i, j] > 0.1:
                #print(vs_img[i,j], vs_img[i, j] / vmax )
                if vs_img[i, j] / vmax < 0.4:
                    text = ax.text(j, i, str(vs_img[i, j]).replace('0.', '.'), ha="center", va="center", color="gray")
                else:
                    text = ax.text(j, i, str(vs_img[i, j]).replace('0.', '.'), ha="center", va="center", color="w")
    if mode == 'graph':
        ax.set_title("Proof history on {} dataset (NEUMANN)".format(dataset), fontsize=18)
    elif mode == 'tensor':
        ax.set_title("Proof history on {} dataset (Tensor-based Reasoner)".format(dataset), fontsize=18)
    fig.tight_layout()
    plt.show()
    folder_path = "plot"
    # plt.savefig(f"{folder_path}/{dataset}_infer_history.svg")
    plt.savefig(f"{folder_path}/{dataset}_{infer_step}_{mode}_history.svg")
    plt.savefig(f"{folder_path}/{dataset}_{infer_step}_{mode}_history.png")

def plot_reasoning_graph(path, reasoning_graph_module):
    #pp = PdfPages(path)

    G = reasoning_graph_module.networkx_graph
    fig = plt.figure(1, figsize=(30, 30))

    first_partition_nodes = list(range(len(reasoning_graph_module.facts)))
    edges = G.edges()
    colors_rg = [G[u][v]['color'] for u, v in edges]
    colors = []
    for c in colors_rg:
        if c == 'r':
            colors.append('indianred')
        elif c == 'b':
            colors.append('royalblue')
    #weights = [G[u][v]['weight'] for u,v in edges]

    nx.draw_networkx(
        G,
        alpha=0.5,
        labels=reasoning_graph_module.node_labels, 
        node_size=2, node_color='lightgray', edge_color=colors, font_size=10,
        pos=nx.drawing.layout.bipartite_layout(G, first_partition_nodes))  # Or whatever other display options you like
    plt.tight_layout()

    plt.show()
    plt.savefig(path)
    plt.close()
    #pp.savefig(fig)
    #pp.close()