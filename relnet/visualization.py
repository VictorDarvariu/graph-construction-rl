import platform

import matplotlib as mpl

from relnet.agent.baseline.baseline_agent import *
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.supervised.sl_agent import SLAgent
from relnet.objective_functions.objective_functions import *
from relnet.state.network_generators import *

if platform.system() == 'Darwin':
    mpl.use("TkAgg")
import matplotlib.animation
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from itertools import product

agent_display_names = {RandomAgent.algorithm_name: "Random",
                       GreedyAgent.algorithm_name: "Greedy",
                       RNetDQNAgent.algorithm_name: "RNet–DQN",
                       SLAgent.algorithm_name: "SL",

                       LowestToLowestDegreeAgent.algorithm_name: "L2L",
                       LowestToRandomDegreeAgent.algorithm_name: "L2R",
                       LowestDegreeProductAgent.algorithm_name: "LDP",
                       LowestBetweennessProductAgent.algorithm_name: "LBP",

                       FiedlerVectorAgent.algorithm_name: "FV",
                       EffectiveResistanceAgent.algorithm_name: "ERes",

                       }

objective_function_display_names = {CriticalFractionRandom.name: '$\mathcal{F}_{random}$',
                                    CriticalFractionTargeted.name: '$\mathcal{F}_{targeted}$',
                                    }

network_generator_display_names = {GNMNetworkGenerator.name: "Erdős–Rényi",
                                   BANetworkGenerator.name: "Barabási–Albert",

                                   EuroroadNetworkGenerator.name: "Euroroad",
                                   ScigridNetworkGenerator.name: "Scigrid",
                                   }

plot_objective_display_names = {"ms_per_move": "Mean milliseconds per move",
                                "cummulative_reward": "Cumulative reward"
                                }

plot_objective_legend_positions = {"ms_per_move": "upper right",
                                   "cummulative_reward": "upper right"
                                   }

use_latex = True
fig_dpi = 200



def animate_episode(state_history, show_animation=True, save_animation=False, animation_filename='animation.htm'):
    fig, ax = plt.subplots(figsize=(8, 8), clear=True)

    def update(i):
        ax.clear()
        state = state_history[i]
        state.display(ax)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(state_history), interval=1000, repeat=True)
    if save_animation:
        # HTML(ani.to_jshtml())
        ani.save(animation_filename, writer='html', fps=1)
    if show_animation:
        plt.show()
    return ani

def set_latex_if_required():
    if use_latex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.unicode'] = True

def plot_eval_histories(results_df,
                      figure_save_path,
                      save_fig=True):

    sns.set(font_scale=1.75)
    plt.rcParams["lines.linewidth"] = 2
    plt.rc('font', family='serif')
    set_latex_if_required()


    #dims = (16.54, 24.81)
    #dims = (16.54, 16.54)
    #dims = (16.54, 12.405)
    #dims = (16.54, 12.405)
    #dims = (16.54, 9.924)
    dims = (16.54, 8.27)

    # legend_i, legend_j = 0, 1
    legend_i, legend_j = -1, -1
    legend_ax = None


    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()
    settings = list(product(perfs, net_types))
    lengths = results_df["L"].unique()

    num_algs = len(results_df["algorithm"].unique())

    num_lengths = len(lengths)
    num_settings = len(settings)

    fig, axes = plt.subplots(num_lengths, num_settings, sharex='row', sharey='row', figsize=dims, squeeze=False)

    for i in range(num_lengths):
        for j in range(num_settings):
            length = lengths[i]
            perf_measure, net_type = settings[j]

            filtered_data = results_df[(results_df['L'] == length) &
                                       (results_df['objective_function'] == perf_measure) &
                                       (results_df['network_generator'] == net_type)]

            filtered_data = filtered_data.rename(columns={"network_size": "$|V|$",
                                                          "value": "Value"})
            if num_algs > 1:
                filtered_data.replace(agent_display_names, inplace=True)

            ax = axes[i][j]
            ax = sns.lineplot(data=filtered_data, x="timestep", y="perf", ax=ax, hue="algorithm")

            ax.set_ylabel('$\mathbf{G}^{validate}$ performance', size="small")
            # ax.set_ylabel('')
            max_steps_used = next(iter(set(filtered_data["max_steps_used"])))
            ax.set_xticks(list(range(0, max_steps_used + int(max_steps_used / 10), int(max_steps_used / 5))))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

            if num_algs > 1:
                ax.legend_.remove()

                handles, labels = ax.get_legend_handles_labels()
                if i == legend_i and j == legend_j:
                    legend_ax = ax

    pad = 2.5  # in points
    rows = lengths
    cols = settings

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{objective_function_display_names[col[0]]}, {network_generator_display_names[col[1]]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')


    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"L = {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    if num_algs > 1:
        if legend_ax is not None:
            legend_ax.legend(handles[0:num_algs + 1], labels, loc='upper right', borderaxespad=0.1,
                             fontsize="small")

    if save_fig:
        fig.tight_layout()
        fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    # plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0



def plot_eval_histories_real_world(results_df,
                      figure_save_path,
                      save_fig=True):

    sns.set(font_scale=1.75)
    plt.rcParams["lines.linewidth"] = 2
    plt.rc('font', family='serif')
    set_latex_if_required()


    #dims = (16.54, 24.81)
    #dims = (16.54, 16.54)
    #dims = (16.54, 12.405)
    #dims = (16.54, 12.405)
    dims = (16.54, 9.924)
    #dims = (16.54, 8.27)

    perfs = list(results_df["objective_function"].unique())
    num_perfs = len(perfs)

    fig, axes = plt.subplots(1, num_perfs, sharex='row', sharey='row', figsize=dims, squeeze=False)

    for i in range(num_perfs):
        perf_measure = perfs[i]

        filtered_data = results_df[results_df['objective_function'] == perf_measure]

        ax = axes[0][i]
        ax = sns.lineplot(data=filtered_data, x="timestep", y="perf", ax=ax)

        # ax.set_ylabel('$\mathbf{G}^{validate}$ performance', size="small")
        ax.set_ylabel('')
        max_steps_used = next(iter(set(filtered_data["max_steps_used"])))
        ax.set_xticks(list(range(0, max_steps_used + int(max_steps_used / 10), int(max_steps_used / 5))))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    pad = 2.5  # in points

    cols = perfs
    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{objective_function_display_names[col]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')


    # for ax, row in zip(axes[:, 0], rows):
    #     ax.annotate(f"L = {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
    #                 rotation=90,
    #                 xycoords=ax.yaxis.label, textcoords='offset points',
    #                 size='medium', ha='right', va='center')

    if save_fig:
        fig.tight_layout()
        fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    # plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0

def plot_size_based_results(results_df,
                      figure_save_path,
                      network_sizes):

    sns.set(font_scale=1.75)
    plt.rcParams["lines.linewidth"] = 2
    plt.rc('font', family='serif')
    set_latex_if_required()

    #dims = (16.54, 24.81)
    #dims = (16.54, 16.54)
    dims = (16.54, 12.405)

    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()
    settings = list(product(perfs, net_types))
    lengths = results_df["tau"].unique()

    num_lengths = len(lengths)
    num_settings = len(settings)

    print(f"lengths, settings are {num_lengths, num_settings}")

    fig, axes = plt.subplots(num_lengths, num_settings, sharex='row', sharey='row', figsize=dims)

    legend_i, legend_j = 0, 0

    for i in range(num_lengths):
        for j in range(num_settings):
            length = lengths[i]
            perf_measure, net_type = settings[j]

            filtered_data = results_df[(results_df['tau'] == length) &
                                       (results_df['objective_function'] == perf_measure) &
                                       (results_df['network_generator'] == net_type)]

            filtered_data = filtered_data.rename(columns={"network_size": "$|V|$",
                                                          "cummulative_reward": "Cumulative reward"})

            ax = axes[i][j]
            ax = sns.lineplot(data=filtered_data, x="$|V|$", y="Cumulative reward", hue="algorithm", ax=ax)
            ax.set_ylabel('$\mathbf{G}^{test}$ performance', size="small")

            ax.legend_.remove()

            handles, labels = ax.get_legend_handles_labels()
            if i == legend_i and j == legend_j:
                legend_ax = ax

            #ax.set_xticks(network_sizes)
            ax.set_xticks([s for s in network_sizes if s % 20 == 0])

    num_algs = len(set(results_df['algorithm']))
    original_labels = labels[1:num_algs + 1]

    display_labels = [agent_display_names[label] for label in original_labels]
    legend_ax.legend(handles=handles[1:], labels=display_labels, loc='upper left', borderaxespad=0.1, fontsize="x-small")


    pad = 2.5  # in points

    rows = lengths
    cols = settings

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{objective_function_display_names[col[0]]}, {network_generator_display_names[col[1]]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(r"$\tau$" + f" = {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    #fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    #plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def plot_size_based_properties(results_df,
                      figure_save_path,
                      network_sizes):

    sns.set(font_scale=1.75)
    plt.rcParams["lines.linewidth"] = 2
    plt.rc('font', family='serif')
    set_latex_if_required()

    #dims = (16.54, 24.81)
    #dims = (16.54, 16.54)

    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()

    num_perfs = len(perfs)
    num_net_types = len(net_types)

    dims = (8.27 * num_perfs, 12.405)

    fig, axes = plt.subplots(num_net_types, num_perfs, sharex='all', sharey='col', figsize=dims)

    for i in range(num_net_types):
        for j in range(num_perfs):

            net_type, perf_measure = net_types[i], perfs[j]

            filtered_data = results_df[(results_df['objective_function'] == perf_measure) &
                                       (results_df['network_generator'] == net_type)]

            filtered_data = filtered_data.rename(columns={"network_size": "$|V|$",
                                                          "value": "Value"})

            ax = axes[i][j]
            ax = sns.lineplot(data=filtered_data, x="$|V|$", y="Value", ax=ax)
            # ax.legend_.remove()
            #ax.set_xticks(network_sizes)
            ax.set_xticks([s for s in network_sizes if s % 20 == 0])

    pad = 2.5  # in points

    rows = net_types
    cols = perfs

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{objective_function_display_names[col]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"{network_generator_display_names[row]}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    #fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    #plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def get_annotation_box_width(fig, annotation):
    extent = annotation.get_window_extent(renderer=fig.canvas.get_renderer())
    x0, x1 = extent.x0, extent.x1
    size_pixels = x1 - x0
    size_inches = size_pixels / fig_dpi
    return size_inches


def ci_aggfunc(series):
    std = np.std(series)
    return 1.96 * std
