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

"""ANATRA runs"""
from ibcdfo.pounders.general_h_funs import identity_combine as combinemodels
sys.path.append('anatra/py/')
sys.path.append('minq5/')
from anatra import anatra

tracks_ANATRA = {}
for shots in [1000, 500, 100, 50]: #[50, 100, 500, 1000]:
    prob = QAOA_Max_Cut(G=G, p=depth, shots=shots)

    tracks_ANATRA[shots] = {}

    for seed in tqdm(range(reps), desc="ANATRA reps"):
        np.random.seed(reps)

        problem = solver_inputs(problem=prob)

        gtol = 1e-13 # this will never be reached, but whatever
        delta = 0.1
        Options = {}
        Options["hfun"] = lambda F: F
        Options["combinemodels"] = combinemodels
        Options["printf"] = 1

        objective = lambda x: problem.objective(x, nargout=2) 
        _, _, _, _, _ = anatra(
            objective,
            problem.problem.initialPoint(),
            problem.problem.numberOfVariables(),
            budget,
            gtol,
            delta,
            1,
            -np.inf * np.ones(problem.problem.numberOfVariables()),
            np.inf * np.ones(problem.problem.numberOfVariables()),
            Options=Options
        )

        tracks_ANATRA[shots][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/{name}_{shots}_ANATRA.pickle", "wb")
        pickle.dump(tracks_ANATRA, file)
        file.close()

"""PyBOBYQA runs"""
from skquant.opt import minimize

tracks_PyBobyqa = {}
for shots in [1000, 500, 100, 50]:
    prob = QAOA_Max_Cut(G=G, p=depth, shots=shots)

    tracks_PyBobyqa[shots] = {}

    for seed in tqdm(range(reps), desc="PyBOBYQA reps"):
        np.random.seed(reps)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-100, 100]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="Bobyqa", rhobeg=0.1)

        tracks_PyBobyqa[shots][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/{name}_{shots}_PyBobyqa.pickle", "wb")
        pickle.dump(tracks_PyBobyqa, file)
        file.close()

