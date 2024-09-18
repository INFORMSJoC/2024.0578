import sys, os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pickle
matplotlib.rcParams.update({"font.size": 12})

sys.path.append(os.getcwd())
from utils.support import manage_replications, manage_replications_others

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
markers = ["o", "^", "X", "*", "s"]

"""problem selection"""
import networkx as nx
import yaml

if sys.argv[1] == "0":
    G = nx.Graph()
    G.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])
    name = "toy"
elif sys.argv[1] == "1":
    G = nx.chvatal_graph()
    name = "chvatal"
elif sys.argv[1] == "2":
    G = nx.binomial_graph(n=6, p=0.6, seed=100)
    name = "erdosh6"
elif sys.argv[1] == "3":
    G = nx.binomial_graph(n=8, p=0.6, seed=100)
    name = "erdosh8"
elif sys.argv[1] == "4":
    G = nx.binomial_graph(n=10, p=0.6, seed=100)
    name = "erdosh10"
elif sys.argv[1] == "5":
    G = nx.binomial_graph(n=12, p=0.6, seed=100)
    name = "erdosh12"
else:
    raise Exception("graph option not available")

file = open("graph_data.yml", "r")
graph_properties = yaml.safe_load(file)[name]
file.close()

depth = 5
evals_to_show = 50 * (2 * depth + 1)
location = "Pickle_files"

grid_cases = [
    [
        {"name": name, "shots": 1000, "factor": 0.5, "eval_lim": evals_to_show},
        {"name": name, "shots": 500, "factor": 0.5, "eval_lim": evals_to_show},
    ],
    [
        {"name": name, "shots": 100, "factor": 0.5, "eval_lim": evals_to_show},
        {"name": name, "shots": 50, "factor": 0.5, "eval_lim": evals_to_show},
    ],
]
for index1, row_case in enumerate(grid_cases):
    for index2, case in enumerate(row_case):
        file = f"{case['name']}_{case['shots']}"

        ax[index1, index2].set_xlim((0, case["eval_lim"]))
        # ax[index1, index2].set_ylim((-graph_properties["cost"], 0))
        if index1 == 0 and index2 == 0:
            ax[index1, index2].set_xlabel("Function evaluations")
            ax[index1, index2].set_ylabel("True value")
        ax[index1, index2].grid(True)
        ax[index1, index2].set_title("Noise level = {}".format(case["shots"]))

        """plot Pybobyqa"""
        Pybobyqa = open(f"{location}/{file}_PyBobyqa.pickle", "rb")
        tracks_Pybobyqa = pickle.load(Pybobyqa)
        tracks_Pybobyqa = tracks_Pybobyqa[case["shots"]]
        quants_Pybobyqa, tracks_Pybobyqa = manage_replications_others(evals_to_show, tracks_Pybobyqa, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), quants_Pybobyqa["best_vals"]["0.5"], color=colors[0], label="PyBOBYQA")

        """plot Imfil"""
        ImFil = open(f"{location}/{file}_ImFil.pickle", "rb")
        tracks_ImFil = pickle.load(ImFil)
        tracks_ImFil = tracks_ImFil[case["shots"]]

        quants_ImFil, tracks_ImFil = manage_replications_others(500, tracks_ImFil, ["best_vals"])

        ax[index1, index2].plot(np.arange(500), quants_ImFil["best_vals"]["0.5"], color=colors[1], label="ImFil")

        """plot anatra"""
        anatra = open(f"{location}/{file}_ANATRA.pickle", "rb")
        tracks_anatra = pickle.load(anatra)
        tracks_anatra = tracks_anatra[case["shots"]]
        quants_anatra, tracks_anatra = manage_replications_others(evals_to_show, tracks_anatra, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), quants_anatra["best_vals"]["0.5"], color=colors[2], label="ANATRA")

        """plot SPSA"""
        spsa = open(f"{location}/{file}_SPSA.pickle", "rb")
        tracks_spsa = pickle.load(spsa)
        tracks_spsa = tracks_spsa[case["shots"]]
        quants_spsa, tracks_spsa = manage_replications_others(evals_to_show, tracks_spsa, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), quants_spsa["best_vals"]["0.5"], color=colors[3], label="SPSA")

        """plot NOMAD"""
        nomad = open(f"{location}/{file}_NOMAD.pickle", "rb")
        tracks_nomad = pickle.load(nomad)
        tracks_nomad = tracks_nomad[case["shots"]]
        quants_nomad, tracks_nomad = manage_replications_others(evals_to_show, tracks_nomad, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), quants_nomad["best_vals"]["0.5"], color=colors[4],
                                label="NOMAD")

        if index1 == 0 and index2 == 0:
            lines_labels = [ax1.get_legend_handles_labels() for ax1 in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, loc="lower center", ncol=4)

#fig.tight_layout()
#plt.show()
figname = name + "_image.png"
plt.savefig(figname)
