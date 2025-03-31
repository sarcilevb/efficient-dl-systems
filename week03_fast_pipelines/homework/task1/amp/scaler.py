from abc import ABC, ABCMeta, abstractmethod

import torch


class GradientScaler(ABC):
    __metaclass__ = ABCMeta

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss_scaled = loss.to(torch.float32) * self.scaling_factor
        loss_scaled.backward()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data /= self.scaling_factor

    @property
    @abstractmethod
    def scaling_factor(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


class StaticScaler(GradientScaler):
    def __init__(self, optimizer: torch.optim.Optimizer, factor: float):
        super().__init__(optimizer)
        self.factor = factor

    @property
    def scaling_factor(self) -> float:
        return self.factor

    def step(self) -> None:
        self.optimizer.step()


class DynamicScaler(GradientScaler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor_upper_bound: float,
        factor_lower_bound: float,
        growth_factor: float,
        backoff_factor: float,
        growth_interval: int,
    ):
        super().__init__(optimizer)
        self.factor_upper_bound = factor_upper_bound
        self.factor_lower_bound = factor_lower_bound
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        self.factor = self.factor_upper_bound / 128
        self.num_steps_wo_overflow = 0

    @property
    def scaling_factor(self) -> float:
        return self.factor

    def step(self) -> None:
        if self._overflow():
            self.factor = max(
                self.factor * self.backoff_factor, self.factor_lower_bound
            )
            self.num_steps_wo_overflow = 0
            return

        self.optimizer.step()

        self.num_steps_wo_overflow += 1
        if self.num_steps_wo_overflow >= self.growth_interval:
            self.factor = min(self.factor * self.growth_factor, self.factor_upper_bound)

    def _overflow(self) -> bool:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and (
                    p.grad.data.isinf().any() or p.grad.data.isnan().any()
                ):
                    return True
        return False
