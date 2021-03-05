import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def compute_rt(nb_evaluations:np.ndarray ,trajectory: np.ndarray, target: float) -> int:

    rt = np.where(trajectory <= target)[0]

    if len(rt) == 0:
        rt = np.inf
    else:
        rt = nb_evaluations[rt[0]]
    return rt


def get_rt_table(nb_evaluations, trajectories, targets):

    nb_runs = trajectories.shape[0]
    nb_precisions = targets.shape[1]

    rt_table = np.empty((nb_runs, nb_precisions))
    for i in range(nb_runs):
        for j in range(nb_precisions):
            rt_table[i, j] = compute_rt(nb_evaluations[i], trajectories[i], targets[i, j])
    return rt_table

def draw_rt(runtimes, max_evaluations, dim):

    nb_runs = len(runtimes)

    #runtimes = np.random.permutation(runtimes)

    # Check if all infinity:
    if np.isinf(runtimes).all():
        return np.inf

    final_rt = 0

    curr_idxs = list(range(nb_runs))
    rt = None
    while True:
        # for i in range(nb_runs):
        # Get an runtime:
        idx = np.random.randint(len(curr_idxs))  # if rt is not None else (nb_bootstrap % nb_runs) + 1
        rt = runtimes[idx]

        # Add runtime:
        final_rt += rt if rt != np.inf else max_evaluations

        # Break if a runtime was found:
        if rt != np.inf:
            break

    return final_rt

def get_simulated_rt_table(rt_table, nb_bootstrap, max_evaluations, dim):

    # Internals:
    nb_runs = rt_table.shape[0]
    nb_precisions = rt_table.shape[1]

    # Getting the simulated rts for each task
    simulated_rt_table = np.empty((nb_bootstrap, nb_precisions))
    for i in range(nb_precisions):
        for j in range(nb_bootstrap):
            simulated_rt_table[j, i] = draw_rt(rt_table[:, i], max_evaluations, dim)

    return simulated_rt_table

def side_names(lines):

    handles = lines

    handles_with_legend = [h for h in handles if not plt.getp(h[-1], 'label').startswith('_line')]
    label_list = [plt.getp(h[-1], 'label') for h in handles_with_legend]

    reslabels = []
    reshandles = []
    ys = {}
    lh = 0

    a, b = plt.xlim()
    maxval = b
    for h in handles_with_legend:
        x2 = []
        y2 = []
        for i in h:
            x2.append(plt.getp(i, "xdata"))
            y2.append(plt.getp(i, "ydata"))

        x2 = np.array(np.hstack(x2))
        y2 = np.array(np.hstack(y2))
        tmp = np.argsort(x2)
        x2 = x2[tmp]
        y2 = y2[tmp]

        h = h[-1]  # we expect the label to be in the last element of h
        tmp = (x2 <= maxval)
        try:
            x2bis = x2[y2 < y2[tmp][-1]][-1]
        except IndexError:  # there is no data with a y smaller than max(y)
            x2bis = 0.
        ys.setdefault(y2[tmp][-1], {}).setdefault(x2bis, []).append(h)
        lh += 1

    if lh <= 1:
        lh = 2
    fontsize_interp = (20.0 - lh) / 10.0
    if fontsize_interp > 1.0:
        fontsize_interp = 1.0
    if fontsize_interp < 0.0:
        fontsize_interp = 0.0
    fontsize_bounds = [9, 17]
    fontsize = fontsize_bounds[0] + fontsize_interp * (fontsize_bounds[-1] - fontsize_bounds[0])
    fontsize = 8
    i = 0  # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in reversed(sorted(ys[j].keys())):
            # enforce "best" algorithm comes first in case of equality
            tmp = []
            for h in ys[j][k]:
                if "best" in plt.getp(h, 'label'):
                    tmp.insert(0, h)
                else:
                    tmp.append(h)
            tmp.reverse()
            ys[j][k] = tmp

            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line')):
                    y = 0.02 + i * 0.96 / (lh - 1)
                    tmp = {}
                    for attr in ('lw', 'ls', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    tmp['color'] = tmp['markeredgecolor']
                    legx = maxval ** 1.11
                    reshandles.extend(plt.plot((maxval, legx), (j, y), clip_on=False, **tmp))
                    reshandles.append(
                        plt.text(maxval ** (0.02 + 1.11), y,
                                 plt.getp(h, 'label'),
                                 horizontalalignment="left",
                                 verticalalignment="center",
                                 fontsize=fontsize))
                    reslabels.append(plt.getp(h, 'label'))
                    i += 1

    # plt.axvline(x=maxval, color='k')  # Not as efficient?
    reshandles.append(plt.plot((maxval, maxval), (0., 1.), clip_on=False, color='k'))
    reslabels.reverse()
    plt.xlim(None, maxval)

def show_ecdf_graph(ecdfs, alg_names, title=None):

    # Create Figure:
    fig, ax = plt.subplots()

    # Internals:
    colors = ['tab:blue', 'tab:green', 'tab:red']
    linestyles = ['-', '--', ':']

    # Plot ECDF:
    lines = []
    for i, (x, y) in enumerate(ecdfs):
        line = plt.plot(x, y, label=alg_names[i], c=colors[i], linestyle=linestyles[i])
        lines.append(line)

    # Beautify:
    # XAxis:
    ax.set_xscale('log')
    #ax.set_xticks([i for i in range(1, 100, 5)])
    ax.set_xticks([1.e+00, 1.e+02, 1.e+04, 1.e+06])
    ax.set_xticklabels(['0', '2', '4', '6'])
    ax.set_xlim(1.0, 1.e+06)#ax.get_xlim()[1])
    #ax.set_xlim(1.0, 100)#ax.get_xlim()[1])


    # YAxis:
    ax.set_yticks(np.arange(0., 1.001, 0.2))
    ax.set_ylim(-0.0, 1.01)

    # Both:
    ax.tick_params()
    ax.grid(True, linewidth=0.5)

    # Complete horizontal line
    c = ax.get_children()
    for j, i in enumerate(c):
        if isinstance(i, matplotlib.lines.Line2D):

            xdata = i.get_xdata()
            ydata = i.get_ydata()

            xdata[xdata == np.inf] = -1

            idx_max = np.where(xdata == xdata.max())[0][-1]
            ax.hlines(ydata[idx_max], xdata[idx_max], ax.get_xlim()[1] + 999999999999, colors=colors[j], linestyles=linestyles[j])

    # Side names:
    side_names(lines)

    # Text:
    fig.suptitle(f'ECDF of Runtime: {title}')
    #ax.set_title(f'ECDF of Runtime: {title}', fontsize=12)
    ax.set_xlabel('log10(# f-evals / dimension)', fontsize=12)
    ax.set_ylabel('Fraction of function,target pairs', fontsize=12)

    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    # #axins = zoomed_inset_axes(ax, 3, loc=1) # zoom = 6
    # axins = inset_axes(ax, 2.0, 2.0) #bbox_to_anchor=(0, 0, 1, 1)) # zoom = 6

    # #lines = []
    # for i, (x, y) in enumerate(ecdfs):
    #     #line = plt.plot(x, y, label=alg_names[i])
    #     axins.plot(x, y)
    #     #lines.append(line)

    # #axins.plot(x, y)
    # axins.set_xlim(1.0, 20.0) # Limit the region for zoom
    # axins.set_ylim(0.0, 0.05)

    # #plt.xticks(visible=False)  # Not present ticks
    # #plt.yticks(visible=False)
    # #
    # ## draw a bbox of the region of the inset axes in the parent axes and
    # ## connecting lines between the bbox and the inset axes area
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Show Image:(0, 0, 0.99, 1)
    layout_rect = (0, 0, 0.99, 0.95)#[0.05, 0.05, 1, 0.95]
    plt.tight_layout(rect=layout_rect)
    plt.savefig("ecdf.pdf", format='pdf')
