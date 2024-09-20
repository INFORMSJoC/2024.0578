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
name = 'Quadratic'
n = int(sys.argv[1])
noise_type = str(sys.argv[2])
evals_to_show = 25 * (n + 1)
location = "Pickle_files"

grid_cases = [
    [
        {"name": name, "noise": 1e-5, "factor": 0.5, "eval_lim": evals_to_show},
        {"name": name, "noise": 1e-3, "factor": 0.5, "eval_lim": evals_to_show},
    ],
    [
        {"name": name, "noise": 1e-1, "factor": 0.5, "eval_lim": evals_to_show},
        {"name": name, "noise": 1e-1, "factor": 0.5, "eval_lim": evals_to_show},
    ],
]
for index1, row_case in enumerate(grid_cases):
    for index2, case in enumerate(row_case):
        file = f"{case['name']}_{noise_type}_{n}_{case['noise']}"

        ax[index1, index2].set_xlim((0, case["eval_lim"]))
        # ax[index1, index2].set_ylim((-graph_properties["cost"], 0))
        if index1 == 0 and index2 == 0:
            ax[index1, index2].set_xlabel("Function evaluations")
            ax[index1, index2].set_ylabel("True value")
        ax[index1, index2].grid(True)
        ax[index1, index2].set_title("Noise level = {}".format(case["noise"]))

        """plot Pybobyqa"""
        Pybobyqa = open(f"{location}/{file}_PyBobyqa.pickle", "rb")
        tracks_Pybobyqa = pickle.load(Pybobyqa)
        tracks_Pybobyqa = tracks_Pybobyqa[case["noise"]]
        quants_Pybobyqa, tracks_Pybobyqa = manage_replications_others(evals_to_show, tracks_Pybobyqa, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), np.log10(quants_Pybobyqa["best_vals"]["0.5"]), color=colors[0], label="PyBOBYQA")

        """plot Imfil"""
        ImFil = open(f"{location}/{file}_ImFil.pickle", "rb")
        tracks_ImFil = pickle.load(ImFil)
        tracks_ImFil = tracks_ImFil[case["noise"]]

        quants_ImFil, tracks_ImFil = manage_replications_others(500, tracks_ImFil, ["best_vals"])

        ax[index1, index2].plot(np.arange(500), np.log10(quants_ImFil["best_vals"]["0.5"]), color=colors[1], label="ImFil")

        """plot anatra"""
        anatra = open(f"{location}/{file}_ANATRA.pickle", "rb")
        tracks_anatra = pickle.load(anatra)
        tracks_anatra = tracks_anatra[case["noise"]]
        quants_anatra, tracks_anatra = manage_replications_others(evals_to_show, tracks_anatra, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), np.log10(quants_anatra["best_vals"]["0.5"]), color=colors[2], label="ANATRA")

        """plot SPSA"""
        spsa = open(f"{location}/{file}_SPSA.pickle", "rb")
        tracks_spsa = pickle.load(spsa)
        tracks_spsa = tracks_spsa[case["noise"]]
        quants_spsa, tracks_spsa = manage_replications_others(evals_to_show, tracks_spsa, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), np.log10(quants_spsa["best_vals"]["0.5"]), color=colors[3], label="SPSA")

        """plot NOMAD"""
        nomad = open(f"{location}/{file}_NOMAD.pickle", "rb")
        tracks_nomad = pickle.load(nomad)
        tracks_nomad = tracks_nomad[case["noise"]]
        quants_nomad, tracks_nomad = manage_replications_others(evals_to_show, tracks_nomad, ["best_vals"])

        ax[index1, index2].plot(np.arange(evals_to_show), np.log10(quants_nomad["best_vals"]["0.5"]), color=colors[4],
                                label="NOMAD")

        if index1 == 0 and index2 == 0:
            lines_labels = [ax1.get_legend_handles_labels() for ax1 in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, loc="lower center", ncol=4)

#fig.tight_layout()
#plt.show()
figname = 'quadratic_' + noise_type + '_' + str(n) + '_image.png'
plt.savefig(figname)
