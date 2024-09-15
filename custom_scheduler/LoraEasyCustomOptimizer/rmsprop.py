import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_


class RMSProp(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        beta (float, optional):
            coefficients used for computing running averages of
            gradient and its square (default: 0.999).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=0.999,
        eps=1e-8,
        weight_decay=0,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data)

                # unpack
                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema_squared = state["ema_squared"].to(torch.float32)
                else:
                    grad = grad.data
                    ema_squared = state["ema_squared"]

                beta = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization))

                # bias correction step size
                bias_correction_sqrt = (1 - beta ** state["step"]) ** (1 / 2)
                step_size = lr

                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    # Perform stepweight decay
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32.data.mul_(1 - step_size * weight_decay)
                    else:
                        p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.addcdiv_(grad, denom, value=-step_size)
                else:
                    p.data.addcdiv_(grad, denom, value=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        return loss
