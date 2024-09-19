from Problems.Problem_class import Problem
import numpy as np

from .QAOA_sim_files.Sampled import Sampled_Simulator, Sampled_Simulator_dist
from .QAOA_sim_files.State_vector import State_Vector_Simulator


class QAOA_Max_Cut(Problem):
    # Functions returning problem parameters
    """
    Inputs :
    p       :   [int] The depth of QAOA circuit
    G       :   [networkx obj]  Graph for max cut
    shots   :   [int, optional] Number of shots done for each sampled evaluation (default 1024)
    seed    :   [int, optional] Seed for simulator"""

    def __init__(self, p, G, shots=1024) -> None:
        self.p = p
        self.n = 2 * p
        self.Graph = G
        self.shots = shots

    def initialPoint(self) -> np.ndarray:
        x = np.zeros([self.n])  # might need to be changed
        return x

    def name(self) -> str:
        s = "QAOA Max Cut"
        return s

    def numberOfConstraintsEqualities(self) -> int:
        return 0

    def numberOfConstraintsInequalities(self) -> int:
        return 0

    def numberOfVariables(self) -> int:
        return self.n

    def evaluateObjectiveFunction(self, x) -> float:
        # Calls the sample Simulator to get a sample by running quantum simulator
        mean, var = Sampled_Simulator(self.Graph, self.p, x, self.shots)
        return mean, var

    def evaluateDistribution(self, backend, x):
        mean, var, obj_dict = Sampled_Simulator_dist(backend, self.Graph, self.p, x, self.shots)

        return mean, var, obj_dict

    def function(self, x) -> float:
        """Calls the state vector simulation for actual value"""
        """Expensive to evaluate"""
        val = State_Vector_Simulator(self.Graph, self.p, x)
        return val

    def clean_up(self) -> None:
        pass

    def noise_value(self, x) -> float:
        raise ("No noise evaluation available")
