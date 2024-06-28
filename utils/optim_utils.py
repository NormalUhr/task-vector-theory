from contextlib import contextmanager

import torch
from torch.optim import Optimizer

__all__ = ['LARS']


class LARS(Optimizer):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.
    __ : https://arxiv.org/abs/1708.03888
    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::
    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate
    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    """

    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        if eps < 0.0:
            raise ValueError('invalid epsilon value: , %f' % eps)
        if trust_coef < 0.0:
            raise ValueError("invalid trust coefficient: %f" % trust_coef)

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef
        self.adaptive_lr = torch.ones([])

    def __getstate__(self):
        lars_dict = {}
        lars_dict['eps'] = self.eps
        lars_dict['trust_coef'] = self.trust_coef
        lars_dict['adaptive_lr'] = self.adaptive_lr
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state

        self.eps = lars_dict['eps']
        self.trust_coef = lars_dict['trust_coef']
        self.adaptive_lr = lars_dict['adaptive_lr']

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    @contextmanager
    def hide_weight_decays(self):
        weight_decays = []

        for group in self.optim.param_groups:
            if 'weight_decay' in group:
                weight_decays.append(group['weight_decay'])
                group['weight_decay'] = 0
            else:
                weight_decays.append(None)

        try:
            yield weight_decays
        finally:
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    continue
                group['weight_decay'] = weight_decay

    def apply_adaptive_lrs(self, weight_decays):
        with torch.no_grad():
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    weight_decay = 0.0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_norm = p.norm()
                    grad_norm = p.grad.norm()

                    # The optimizer class has no method to change `dtype` of
                    # its inner tensors (like `adaptive_lr`) and to select to
                    # use CPU or GPU with Tensor. LARS's interface follows the
                    # optimizer class's interface, so LARS cannot change
                    # `dtype` of inner tensors explicitly also. In that
                    # context, we have constructed LARS can modify its member
                    # variable's spec implicitly by comparing with given spec
                    # by the original optimizer's element.
                    # param_norm_spec = (param_norm.is_cuda, param_norm.type())
                    # adaptive_lr_spec = (self.adaptive_lr.is_cuda, self.adaptive_lr.type())
                    #
                    # if param_norm_spec != adaptive_lr_spec:
                    self.adaptive_lr = torch.ones_like(param_norm)

                    # calculate adaptive lr & weight decay
                    adaptive_lr = self.compute_adaptive_lr(
                        param_norm,
                        grad_norm,
                        weight_decay,
                        # self.eps,
                        # self.trust_coef,
                        # self.adaptive_lr
                    )

                    p.grad.add_(weight_decay, p.data)
                    p.grad.mul_(adaptive_lr)

    def step(self, *args, **kwargs):
        with self.hide_weight_decays() as weight_decays:
            self.apply_adaptive_lrs(weight_decays)
            return self.optim.step(*args, **kwargs)

    def compute_adaptive_lr(self, param_norm, grad_norm, weight_decay):

        if param_norm > 0 and grad_norm > 0:
            divisor = grad_norm + weight_decay * param_norm + self.eps
            self.adaptive_lr = param_norm / divisor * self.trust_coef
        else:
            self.adaptive_lr = 1.0
        return self.adaptive_lr


# Source code from https://github.com/cybertronai/pytorch-lamb
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
