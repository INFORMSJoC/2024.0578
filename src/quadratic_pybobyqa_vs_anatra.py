import sys, os
import numpy as np
import pickle

sys.path.append(os.getcwd())

from tqdm import tqdm

from Problems.Quadratic import Quadratic
n = int(sys.argv[1])
noise_type = str(sys.argv[2])
budget = 25 * (n + 1)
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
        true_val = self.problem.function(x)

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
for noise in [1e-5,1e-3,1e-1]:
    prob = Quadratic(n=n, noise_level=noise, noise_type=noise_type)

    tracks_ANATRA[noise] = {}

    for seed in tqdm(range(reps), desc="ANATRA reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        gtol = 1e-13 # this will never be reached, but whatever
        delta = 0.1
        Options = {}
        Options["hfun"] = lambda F: F
        Options["combinemodels"] = combinemodels
        Options["printf"] = 1

        if noise_type == 'gaussian':
            eps_bound = None
        elif noise_type == 'uniform':
            eps_bound = noise

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
            eps_bound=eps_bound,
            Options=Options
        )

        tracks_ANATRA[noise][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/Quadratic_{noise_type}_{n}_{noise}_ANATRA.pickle", "wb")
        pickle.dump(tracks_ANATRA, file)
        file.close()

"""PyBOBYQA runs"""
from skquant.opt import minimize

tracks_PyBobyqa = {}
for noise in [1e-5,1e-3,1e-1]:
    prob = Quadratic(n=n, noise_level=noise)

    tracks_PyBobyqa[noise] = {}

    for seed in tqdm(range(reps), desc="PyBOBYQA reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-np.pi, np.pi]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="Bobyqa", rhobeg=0.1)

        tracks_PyBobyqa[noise][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

    file = open(f"Pickle_files/Quadratic_{noise_type}_{n}_{noise}_PyBobyqa.pickle", "wb")
    pickle.dump(tracks_PyBobyqa, file)
    file.close()
