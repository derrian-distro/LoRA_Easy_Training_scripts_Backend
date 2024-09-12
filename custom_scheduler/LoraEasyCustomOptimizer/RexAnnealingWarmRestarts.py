from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class RexAnnealingWarmRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        cycle_multiplier: float = 1,
        first_cycle_max_steps: int = 1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        d: float = 0.9,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.cycle_multiplier = cycle_multiplier
        self.gamma = gamma  # debating calling this decay_rate or something
        self.last_epoch = last_epoch
        self.d = d

        # new run
        if last_epoch == -1:
            if warmup_steps >= first_cycle_max_steps:
                raise ValueError(
                    f"[-] warmup_steps must be smaller than first_cycle_max_steps. "
                    f"{warmup_steps} < {first_cycle_max_steps}"
                )
            self.setup_optimizer(warmup_steps, first_cycle_max_steps, min_lr)
        self.validate_optimizer()

        self._initial_step()

    def setup_optimizer(
        self,
        warmup_steps: int,
        first_cycle_max_steps: int,
        min_lr: float,
    ) -> Optimizer:
        for group in self.optimizer.param_groups:
            if "warmup_steps" not in group:
                group.setdefault("warmup_steps", warmup_steps)
            if "current_cycle_max_steps" not in group:
                group.setdefault("current_cycle_max_steps", first_cycle_max_steps)
            if "min_lr" not in group:
                group.setdefault("min_lr", min_lr if min_lr < group["lr"] else 0)
            group.setdefault("current_cycle", 0)
            group.setdefault("current_cycle_step", -1)
            group.setdefault("initial_lr", group["lr"])
            group.setdefault("current_max_lr", group["lr"])

    def validate_optimizer(self):
        for i, group in enumerate(self.optimizer.param_groups):
            for key in {
                "warmup_steps",
                "current_cycle_max_steps",
                "min_lr",
                "current_cycle",
                "current_cycle_step",
                "initial_lr",
                "current_max_lr",
            }:
                if key not in group:
                    raise KeyError(
                        f"param '{key}' is not specified in param_groups[{i}] when resuming an optimizer"
                    )
            if group["warmup_steps"] >= group["current_cycle_max_steps"]:
                raise ValueError(
                    f"[-] warmup_steps must be smaller than first_cycle_max_steps. "
                    f"{group['warmup_steps']} < {group['current_cycle_max_steps']}"
                )

    def _calc_first_step(self, group: list[float | int]):
        while group["current_cycle_step"] >= group["current_cycle_max_steps"]:
            group = self._update_cycle(group)
        return group

    def _update_step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            if group["current_cycle_step"] == -1:
                group = self._calc_first_step(group)
                self.optimizer.param_groups[i] = group
            group["current_cycle_step"] += 1
            group = self._update_cycle(group)
            self.optimizer.param_groups[i] = group

    def _update_cycle(self, group: list[float | int]):
        if group["current_cycle_step"] < group["current_cycle_max_steps"]:
            return group
        group["current_cycle_step"] -= group["current_cycle_max_steps"]
        group["current_cycle"] += 1
        group["current_cycle_max_steps"] = (
            round((group["current_cycle_max_steps"] - group["warmup_steps"]) * self.cycle_multiplier)
            + group["warmup_steps"]
        )
        group["current_max_lr"] = group["initial_lr"] * (self.gamma ** group["current_cycle"])
        return group

    def get_lr(self) -> float:
        self._update_step()
        lrs = []
        for group in self.optimizer.param_groups:
            if group["current_max_lr"] <= group["min_lr"]:
                lrs.append(group["min_lr"])
                continue
            lr_range = group["current_max_lr"] - group["min_lr"]
            if group["current_cycle_step"] < group["warmup_steps"]:
                lrs.append(lr_range * group["current_cycle_step"] / group["warmup_steps"] + group["min_lr"])
                continue
            normalized_step = group["current_cycle_step"] - group["warmup_steps"]
            normalized_max_steps = group["current_cycle_max_steps"] - group["warmup_steps"]
            progress = normalized_step / normalized_max_steps
            divider = (1 - self.d) + (self.d * (1 - progress))
            lrs.append(group["min_lr"] + lr_range * ((1 - progress) / divider))
        return lrs
