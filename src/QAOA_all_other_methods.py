import sys, os
import numpy as np
import pickle

sys.path.append(os.getcwd())

from tqdm import tqdm

# generate QAOA problem
from Problems.QAOA_max_cut import QAOA_Max_Cut
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
budget = 50 * (2 * depth + 1)
reps = 30

class solver_inputs:
    def __init__(self, problem) -> None:
        self.problem = problem
        self.evals = 0
        self.best_val = np.inf

        self.evals_list = []
        self.best_list = []

    def objective(self, x, nargout=1):  #
        mean, var = self.problem.evaluateObjectiveFunction(x)
        true_val, _ = self.problem.function(x)

        self.evals += 1
        self.evals_list.append(self.evals)

        if true_val < self.best_val:
            self.best_val = true_val

        self.best_list.append(self.best_val)

        if nargout == 1:
            return mean
        elif nargout == 2:
            return mean, var

"""SPSA runs"""

tracks_SPSA = {}
from qiskit.algorithms.optimizers import SPSA

for shots in [1000, 500, 100, 50]:
    prob = QAOA_Max_Cut(G=G, p=depth, shots=shots)

    tracks_SPSA[shots] = {}

    for seed in tqdm(range(reps), desc="SPSA reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        spsa_object = SPSA(maxiter=budget)
        spsa_result = spsa_object.optimize(problem.problem.n, problem.objective, initial_point=problem.problem.initialPoint())

        tracks_SPSA[shots][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/{name}_{shots}_SPSA.pickle", "wb")
        pickle.dump(tracks_SPSA, file)
        file.close()

"""ImFil runs"""

tracks_ImFil = {}
from skquant.opt import minimize
for shots in [1000, 500, 100, 50]:
    prob = QAOA_Max_Cut(G=G, p=depth, shots=shots)

    tracks_ImFil[shots] = {}

    for seed in tqdm(range(reps), desc="ImFil reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-100, 100]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="ImFil")

        tracks_ImFil[shots][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/{name}_{shots}_ImFil.pickle", "wb")
        pickle.dump(tracks_ImFil, file)
        file.close()

"""NOMAD runs"""

tracks_NOMAD = {}
from skquant.opt import minimize
for shots in [1000, 500, 100, 50]:
    prob = QAOA_Max_Cut(G=G, p=depth, shots=shots)

    tracks_NOMAD[shots] = {}

    for seed in tqdm(range(reps), desc="NOMAD reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-100, 100]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="NOMAD")

        tracks_NOMAD[shots][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/{name}_{shots}_NOMAD.pickle", "wb")
        pickle.dump(tracks_NOMAD, file)
        file.close()
