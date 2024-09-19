from curses import COLOR_RED
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.gridspec as gridspec

def draw(pdag, colored_set=set(), solved_set=set(), affected_set=set(), nw_ax=None, edge_weights=None, savefile=None):
    """ 
    plot a partially directed graph
    """
    plt.clf()

    p = pdag.nnodes

    if nw_ax is None:
        nw_ax = plt.subplot2grid((4, 4), (0, 0), colspan=12, rowspan=12)

    plt.gcf().set_size_inches(4, 4)

    # directed edges
    d = nx.DiGraph()
    d.add_nodes_from(list(range(p)))
    for (i, j) in pdag.arcs:
        d.add_edge(i, j)

    # undirected edges
    e = nx.Graph()
    try:
        for pair in pdag.edges:
            (i, j) = tuple(pair)
            e.add_edge(i, j)
    except:
        print('there are no undirected edges')
    
    # edge color
    if edge_weights is not None:
        color_d = []
        for i,j in d.edges:
            color_d.append(abs(edge_weights[i,j]))

        color_e = []
        for i,j in e.edges:
            color_e.append(abs(edge_weights[i, j]))
    else:
        color_d = 'k'
        color_e = 'k'


    # plot
    print("plotting...")
    pos = nx.circular_layout(d)
    nx.draw(e, pos=pos, node_color='w', style = 'dashed', ax=nw_ax, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1, edge_color=color_e)
    color = ['w']*p
    for i in affected_set:
        color[i] = 'orange'
    for i in colored_set:
        color[i] = 'y'
    for i in solved_set:
        color[i] = 'grey'
    nx.draw(d, pos=pos, node_color=color, ax=nw_ax, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1, edge_color=color_d) #, edge_width=2)
    nx.draw_networkx_labels(d, pos, labels={node: node for node in range(p)}, ax=nw_ax)

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    plt.close()

plt.rcParams.update({'font.size': 18})

def show_results(U, N, U_estimates, N_estimates):
    num_latent = U.shape[1]
    U_np = U.detach().numpy()
    N_np = N.detach().numpy()

    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(nrows=1, ncols=2)

    gs_left = gridspec.GridSpecFromSubplotSpec(num_latent, num_latent, subplot_spec=gs[0])

    for i in range(num_latent):
        for j in range(num_latent):
            ax = fig.add_subplot(gs_left[i, j])
            ax.scatter(U_np[:, i], U_estimates[:, j], alpha=0.5, c=U_np[:, 0], cmap='PuBu')
            if i == 0:
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                ax.set_xticks([round(min(U_np[:, i]), 1), round(max(U_np[:, i]), 1)])
            else:
                ax.set_xticks([])

            if j == num_latent - 1:
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                ax.set_yticks([round(min(U_estimates[:, j]), 1), round(max(U_estimates[:, j]), 1)])
            else:
                ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(f'Est. Latent {i + 1}', fontsize=25)
            if i == num_latent - 1:
                ax.set_xlabel(f'Latent {j + 1}', fontsize=25)

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax_right = fig.add_subplot(gs[1])

    combined = np.concatenate((N_np, N_estimates), axis=1)
    corr = np.corrcoef(combined, rowvar=False)
    corr_matrix = corr[0:num_latent, num_latent:2*num_latent]
    abs_corr_matrix = np.abs(corr_matrix)

    red_white_blue = LinearSegmentedColormap.from_list('RedWhiteBlue', ['red', 'white', 'cornflowerblue'], N=256)

    sns.heatmap(
        abs_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='PuBu',
        ax=ax_right,
        cbar=False,
        xticklabels=[f'Est. Noise {i + 1}' for i in range(num_latent)],
        yticklabels=[f'Noise {i + 1}' for i in range(num_latent)],
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 30} 
    )

    ax_right.tick_params(axis='x', labelsize=25)
    ax_right.tick_params(axis='y', labelsize=25)

    fig.text(0.25, 0.95, 'Latent Variable Estimates', ha='center', fontsize=30)
    fig.text(0.75, 0.95, 'Noise Variable Estimates', ha='center', fontsize=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()