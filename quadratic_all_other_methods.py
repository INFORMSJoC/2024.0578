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

"""SPSA runs"""

tracks_SPSA = {}
from qiskit.algorithms.optimizers import SPSA

for noise in [1e-5,1e-3,1e-1]:
    prob = Quadratic(n=n, noise_level=noise, noise_type=noise_type)

    tracks_SPSA[noise] = {}

    for seed in tqdm(range(reps), desc="SPSA reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        spsa_object = SPSA(maxiter=budget)
        spsa_result = spsa_object.optimize(problem.problem.n, problem.objective, initial_point=problem.problem.initialPoint())

        tracks_SPSA[noise][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/Quadratic_{noise_type}_{n}_{noise}_SPSA.pickle", "wb")
        pickle.dump(tracks_SPSA, file)
        file.close()

"""ImFil runs"""

tracks_ImFil = {}
from skquant.opt import minimize
for noise in [1e-5,1e-3,1e-1]:
    prob = Quadratic(n=n, noise_level=noise, noise_type=noise_type)

    tracks_ImFil[noise] = {}

    for seed in tqdm(range(reps), desc="ImFil reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-np.pi, np.pi]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="ImFil")

        tracks_ImFil[noise][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/Quadratic_{noise_type}_{n}_{noise}_ImFil.pickle", "wb")
        pickle.dump(tracks_ImFil, file)
        file.close()

"""NOMAD runs"""

tracks_NOMAD = {}
from skquant.opt import minimize
for noise in [1e-5,1e-3,1e-1]:
    prob = Quadratic(n=n, noise_level=noise, noise_type=noise_type)

    tracks_NOMAD[noise] = {}

    for seed in tqdm(range(reps), desc="NOMAD reps"):
        np.random.seed(seed)

        problem = solver_inputs(problem=prob)

        bounds = np.array([[-np.pi, np.pi]] * problem.problem.n, dtype=float)
        minimize(problem.objective, problem.problem.initialPoint(), bounds, budget, method="NOMAD")

        tracks_NOMAD[noise][seed] = {"evals": problem.evals_list, "best_vals": problem.best_list}

        file = open(f"Pickle_files/Quadratic_{noise_type}_{n}_{noise}_NOMAD.pickle", "wb")
        pickle.dump(tracks_NOMAD, file)
        file.close()

