from abc import ABC, abstractmethod
import numpy as np

# Class that defines a template to specify problems


class Problem(ABC):

    """
    The functions below allow to access to problem parameters
    """

    def __init__(self) -> None:
        self.observed_evaluations = []
        self.true_evaluations = []
        self.best_observed_evaluation = []
        self.best_true_evaluation = []
        self.numberOfEvaluations = 0
        super().__init__()

    @abstractmethod
    def initialPoint(self) -> np.ndarray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def numberOfConstraintsEqualities(self) -> int:
        pass

    @abstractmethod
    def numberOfConstraintsInequalities(self) -> int:
        pass

    @abstractmethod
    def numberOfVariables(self) -> int:
        pass

    def evaluateObjectiveFunction(self, x) -> float:
        self.numberOfEvaluations += 1

        true = self.function(x)
        noise = self.noise_value(x)

        self.observed_evaluations.append(true + noise)
        self.true_evaluations.append(true)

        if self.numberOfEvaluations > 1:
            if true < self.best_true_evaluation[-1]:
                self.best_true_evaluation.append(true)
            else:
                self.best_true_evaluation.append(self.best_true_evaluation[-1])

            if true + noise < self.best_observed_evaluation[-1]:
                self.best_observed_evaluation.append(true + noise)
            else:
                self.best_observed_evaluation.append(self.best_observed_evaluation[-1])

        else:
            self.best_true_evaluation.append(true)
            self.best_observed_evaluation.append(true + noise)

        return true + noise

    @abstractmethod
    def function(self, x) -> float:
        # different for each problem
        pass

    @abstractmethod
    def noise_value(self, x) -> float:
        # different for each problem
        pass

    def clean_up(self):
        self.observed_evaluations = []
        self.true_evaluations = []
        self.best_observed_evaluation = []
        self.best_true_evaluation = []
        self.numberOfEvaluations = 0

    """
        Functions below act as evaluation oracles
        Each function returns a tuple, (value, err)
        x : decision variable to evaluate on
        yE : dual variable for equality constraints
        yI : dual variable for inequality constraints
        type : gradient type, true (full) or stochastic
        factor : extra parameter, for eg. batch, scale, etc
    """

    # @abstractmethod
    # def evaluateConstraintFunctionEqualities(self, x) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def evaluateConstraintFunctionInequalities(self,x) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def evaluateConstraintJacobianEqualities(self,x) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def evaluateConstraintJacobianInequalities(self, x) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def evaluateHessianOfLagrangian(self, x, yE, yI) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def evaluateObjectiveGradient(self, x, type, factor) -> np.ndarray:
    #     pass
