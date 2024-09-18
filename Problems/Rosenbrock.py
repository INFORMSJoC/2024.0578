from Problems.Problem_class import Problem
import numpy as np


class Rosenbrock(Problem):
    # Functions returning problem parameters

    def __init__(self, n=2, noise_level=0, noise_type='gaussian') -> None:
        super().__init__()

        self.n = n
        self.noise = noise_level
        self.type = noise_type
        if noise_type == 'gaussian':
            self.var = noise_level ** 2
        elif noise_type == 'uniform':
            self.var = noise_level ** 2 / 3.0

    def initialPoint(self) -> np.ndarray:
        x = np.zeros([self.n])
        return x

    def name(self) -> str:
        s = "Rosenbrock"
        return s

    def numberOfConstraintsEqualities(self) -> int:
        return 0

    def numberOfConstraintsInequalities(self) -> int:
        return 0

    def numberOfVariables(self) -> int:
        return self.n

    def function(self, x) -> float:
        val = 0

        for i in range(self.n - 1):
            val = val + 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

        return val

    def noise_value(self, x) -> float:
        if self.type == 'gaussian':
            return np.random.normal(0, self.noise)
        elif self.type == 'uniform':
            return np.random.uniform(-self.noise, self.noise)

    def evaluateObjectiveFunction(self, x) -> float:
        val = self.function(x) + self.noise_value(x)
        var = self.var
        return val, var
