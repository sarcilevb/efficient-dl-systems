from typing import Callable, Dict, List, Type

import torch
import transformers


class Offloading(torch.nn.Module):
    _BYTES_IN_GB = 1024**3

    class OpOffloadingWrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, module: torch.nn.Module, *fwd_args, **fwd_kwargs):
            pass

        @staticmethod
        def backward(ctx, *grad_outputs):
            pass

    class ModuleOffloading(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            self.module = module.named_modules

        def forward(self, *fwd_args, **fwd_kwargs):
            return Offloading.OpOffloadingWrapper.apply(
                self.module, *fwd_args, **fwd_kwargs
            )

    def __init__(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        cuda_device_idx: int,
        max_size_in_gb: int,
        prefetch: bool,
    ):
        super().__init__()
        self.model = model
        self.cuda_device_idx = cuda_device_idx
        self.max_size_in_gb = max_size_in_gb
        self.prefetch = prefetch

        print("Tracing forward module invocation order...")
        self.forward_load_order = self._record_module_loading_order(
            lambda: self.model(sample_input)
        )
        print("Tracing backward module invocation order...")

        def _bwrd():
            inp = sample_input.detach()
            sample_output = self.model(inp)
            self._losss = sample_output.logits.sum()
            self._losss.backward()

        self.backward_load_order = self._record_module_loading_order(_bwrd)

    def _record_module_loading_order(
        self,
        trigger: Callable,
    ) -> List[List[torch.nn.Module]]:
        module_invocation_order_and_param_sizes = []

        class Trace(torch.nn.Module):
            class TraceFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, module: torch.nn.Module, inputs):
                    ctx.module = module
                    Trace.TraceFn.record_trace(module)
                    return inputs

                @staticmethod
                def backward(ctx, grad_output):
                    Trace.TraceFn.record_trace(ctx.module)
                    return None, grad_output

                @staticmethod
                def record_trace(module: torch.nn.Module):
                    param_size_gb = sum(
                        (
                            p.numel() * p.element_size() / Offloading._BYTES_IN_GB
                            for _, p in module.named_parameters(recurse=False)
                        )
                    )
                    param_size_gb += sum(
                        (
                            b.numel() * b.element_size() / Offloading._BYTES_IN_GB
                            for _, b in module.named_buffers(recurse=False)
                        )
                    )
                    if param_size_gb > 0:
                        module_invocation_order_and_param_sizes.append(
                            (module, param_size_gb)
                        )
                    print(module.__class__.__name__, param_size_gb)

            def __init__(self, module: torch.nn.Module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                result = self.module(*args, **kwargs)
                return Trace.TraceFn.apply(self.module, result)

            def __getattr__(self, name):
                # This is called ONLY if attribute not found normally
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    if name == "module":  # prevent recursion
                        raise AttributeError()
                    return getattr(self.module, name)

            def __iter__(self):
                return iter(self.module)

        _wrap_all_submodules(self.model, Trace)

        trigger()

        module_prefetch_groups = []
        current_group = []
        current_size_in_gb = 0.0
        for module, size_in_gb in module_invocation_order_and_param_sizes:
            if current_size_in_gb + size_in_gb > self.max_size_in_gb:
                if len(current_group) > 0:
                    module_prefetch_groups.append(current_group)
                current_group = []
                current_size_in_gb = 0.0
            current_size_in_gb += size_in_gb
            current_group.append(module)
        if len(current_group) > 0:
            module_prefetch_groups.append(current_group)

        _unwrap_all_submodules(self.model)

        return module_prefetch_groups

        # iterate over all modules included nested ones, add pre-hooks and record invocation order, plus count the number of parameters and buffers for each

        # split modules into subgroups based on the number of parameters and buffers (each subgroup will be within X MB of memory) while preserving the invocation order
        # based on this invocation order, do:
        # to handle forward pass, (a) for start of list modules, wrap their forward function in a function that initiates next module list prefetch and saves the future.
        #       (b) for end of list module, await that future after the wrapped forward before returning.
        #       (c) for every module, after forward, offload their params. Note that this offload operation by design will be enqueued before enqueuing next module list prefetch. So by cuda stream semantics it will return first and peak memory usage will be controlled.

    def forward(self, x):
        m = transformers.AutoModelForCausalLM.from_pretrained()
        m.forward()


def _wrap_all_submodules(
    module: torch.nn.Module, wrapper_module_cls: Type[torch.nn.Module]
):
    for name, submodule in module.named_children():
        _wrap_all_submodules(submodule, wrapper_module_cls)
        wrapped_submodule = wrapper_module_cls(submodule)
        setattr(module, name, wrapped_submodule)


def _unwrap_all_submodules(module: torch.nn.Module):
    for name, submodule in module.named_children():
        _unwrap_all_submodules(submodule)
        if submodule.__class__.__name__ == "Trace":
            setattr(module, name, submodule.module)
