from Problems.Problem_class import Problem
import numpy as np


class Quadratic(Problem):
    # Functions returning problem parameters

    def __init__(self, n, A=None, b=None, noise_level=0, noise_type='gaussian') -> None:
        super().__init__()

        self.n = n
        self.noise = noise_level
        self.type = noise_type
        if noise_type == 'gaussian':
            self.var = noise_level ** 2
        elif noise_type == 'uniform':
            self.var = noise_level ** 2 / 3.0

        if A is None:
            self.A = np.identity(self.n)
        else:
            self.A = A

        if b is None:
            self.b = np.zeros([self.n, 1])
        else:
            self.b = b

    def initialPoint(self) -> np.ndarray:
        x = np.ones([self.n]) # note this was previously multiplied by 5
        return x

    def name(self) -> str:
        s = "Quadratic"
        return s

    def numberOfConstraintsEqualities(self) -> int:
        return 0

    def numberOfConstraintsInequalities(self) -> int:
        return 0

    def numberOfVariables(self) -> int:
        return self.n

    def function(self, x) -> float:
        val = np.dot(np.dot(x.T, self.A), x) + np.dot(x.T, self.b)

        return val[0]

    def noise_value(self, x) -> float:
        if self.type == 'gaussian':
            return np.random.normal(0, self.noise)
        elif self.type == 'uniform':
            return np.random.uniform(-self.noise, self.noise)

    def evaluateObjectiveFunction(self, x) -> float:
        val = self.function(x) + self.noise_value(x)
        var = self.var
        return val, var
