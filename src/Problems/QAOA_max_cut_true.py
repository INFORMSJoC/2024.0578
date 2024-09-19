from Problems.Problem_class import Problem
import numpy as np

from .QAOA_sim_files.Sampled import Sampled_Simulator
from .QAOA_sim_files.State_vector import State_Vector_Simulator


class QAOA_Max_True(Problem):
    # Functions returning problem parameters
    """
    Inputs :
    p       :   [int] The depth of QAOA circuit
    G       :   [networkx obj]  Graph for max cut
    seed    :   [int, optional] Seed for simulator"""

    def __init__(self, p, G) -> None:
        self.p = p
        self.n = 2 * p
        self.Graph = G

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
        mean, var = self.function(x)
        return mean, var

    def function(self, x) -> float:
        """Calls the state vector simulation for actual value"""
        """Expensive to evaluate"""
        mean, var = State_Vector_Simulator(self.Graph, self.p, x)
        return mean, var

    def clean_up(self) -> None:
        pass

    def noise_value(self, x) -> float:
        raise ("No noise evaluation available")
