
import warnings

from torch.optim.lr_scheduler import LambdaLR


class NoamLR(LambdaLR):
    """

    Noam learning rate scheduling using torch scheduler.
    Works same as NoamOpt class.
    """

    def __init__(self, model_dim, warmup_step, optimizer, last_epoch=-1, verbose=False):
        self.model_dim = model_dim
        self.warmup_step = warmup_step

        def func(step, warmup_step):
            return min(step ** -0.5, step * warmup_step ** -1.5)

        lr_lambda = func
        super(NoamLR, self).__init__(optimizer, lr_lambda, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        # print(self._step_count)
        # print([lmbda(self._step_count, self.warmup_step)
        #         for lmbda in self.lr_lambdas])
        return [(self.model_dim ** -0.5) * lmbda(self._step_count, self.warmup_step)
                for lmbda in self.lr_lambdas]


class NoamOpt:
    """Optimizer wrapper that implements noam scheduling learning rate."""

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        # self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
