import torch


class Detach(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        detached_args = []
        detached_kwargs = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.detach()
            detached_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            detached_kwargs[k] = v

        return self.module(*detached_args, **detached_kwargs)
