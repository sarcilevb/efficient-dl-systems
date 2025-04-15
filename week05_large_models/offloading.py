from collections import defaultdict
from textwrap import wrap
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
import transformers


class OffloadContext:
    def __init__(self, cuda_device: str):
        self.cuda_device = cuda_device
        self.module_to_fetching_list_forward: Dict[int, List[ModuleOffloading]] = {}
        self.module_to_fetching_list_backward: Dict[int, List[ModuleOffloading]] = {}
        self.cuda_io_stream: torch.cuda.Stream = torch.cuda.Stream()


class ModuleOffloading(torch.nn.Module):
    class PreForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, module: "ModuleOffloading", *args):
            _schedule_prefetch(module, forward=True)
            module.wait_until_loads()
            ctx.module = module
            return args if len(args) > 1 else args[0]

        @staticmethod
        def backward(ctx, *grad_output):
            ctx.module.schedule_offload()
            return None, grad_output

    class PostForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, module: "ModuleOffloading", *args):
            ctx.module = module
            module.schedule_offload()
            return args if len(args) > 1 else args[0]

        @staticmethod
        def backward(ctx, *grad_output):
            _schedule_prefetch(ctx.module, forward=False)
            ctx.module.wait_until_loads()
            return None, grad_output

    def __init__(self, module: torch.nn.Module, ctx: OffloadContext):
        self.module = module
        self.ctx = ctx
        self.load_future: Optional[torch.cuda.Event] = None

    def forward(self, *fwd_args, **fwd_kwargs):
        fwd_args = ModuleOffloading.PreForward.apply(self, *fwd_args)
        output = self.module(*fwd_args, **fwd_kwargs)
        return ModuleOffloading.PostForward.apply(self, output)

    def schedule_load(self):
        load_future = self._move(self.ctx.cuda_device)
        self.load_future = load_future

    def schedule_offload(self):
        self._move("cpu")

    def wait_until_loads(self):
        if self.load_future is not None:
            # pyright: ignore[reportArgumentType]
            torch.cuda.default_stream().wait_event(self.load_future)

    def _move(self, device: str) -> torch.cuda.Event:
        with torch.cuda.stream(self.ctx.cuda_io_stream):
            for parameter in self.module.parameters():
                parameter.data = parameter.data.to(device, non_blocking=True)
            for buffer in self.module.buffers():
                buffer.data = buffer.data.to(device, non_blocking=True)
            # pyright: ignore[reportAssignmentType]
            load_future: torch.cuda.Event = torch.cuda.Event()
            load_future.record(self.ctx.cuda_io_stream)
        return load_future


class Offloading(torch.nn.Module):
    _BYTES_IN_GB = 1024**3

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

        def _run_forward_backward():
            sample_output = self.model(sample_input)
            self._losss = sample_output.logits.sum()
            self._losss.backward()

        print("Tracing module invocation order...")
        forward_load_order, backward_load_order = self._record_module_loading_order(
            _run_forward_backward
        )

        context = OffloadContext(f"cuda:{cuda_device_idx}")
        _wrap_all_submodules(
            self.model, lambda module: ModuleOffloading(module, context)
        )
        self.forward_load_order = self._prepare_load_order(forward_load_order)
        self.backward_load_order = self._prepare_load_order(backward_load_order)
        context.module_to_fetching_list_forward = self.forward_load_order
        context.module_to_fetching_list_backward = self.backward_load_order

    def forward(self, x):
        return self.model(x)

    def _record_module_loading_order(
        self,
        trigger: Callable,
    ) -> Tuple[List[List[torch.nn.Module]], List[List[torch.nn.Module]]]:
        module_invocation_order_and_param_sizes = defaultdict(list)

        class Trace(torch.nn.Module):
            backward_started = False

            class TraceFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, module: torch.nn.Module, inputs):
                    ctx.module = module

                    trace_key = "forward" if not Trace.backward_started else "backward"
                    Trace.TraceFn.record_trace(
                        module, module_invocation_order_and_param_sizes[trace_key]
                    )
                    return inputs

                @staticmethod
                def backward(ctx, grad_output):
                    Trace.backward_started = True
                    Trace.TraceFn.record_trace(
                        ctx.module, module_invocation_order_and_param_sizes["backward"]
                    )
                    return None, grad_output

                @staticmethod
                def record_trace(
                    module: torch.nn.Module, out: List[Tuple[torch.nn.Module, float]]
                ):
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
                        out.append((module, param_size_gb))
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
        _unwrap_all_submodules(self.model)

        return (
            self._split_modules_into_load_chunks(
                module_invocation_order_and_param_sizes["forward"]
            ),
            self._split_modules_into_load_chunks(
                module_invocation_order_and_param_sizes["backward"]
            ),
        )

    def _split_modules_into_load_chunks(
        self, modules_and_sizes: List[Tuple[torch.nn.Module, float]]
    ) -> List[List[torch.nn.Module]]:
        module_prefetch_groups = []
        current_group = []
        current_size_in_gb = 0.0
        for module, size_in_gb in modules_and_sizes:
            if current_size_in_gb + size_in_gb > self.max_size_in_gb:
                if len(current_group) > 0:
                    module_prefetch_groups.append(current_group)
                current_group = []
                current_size_in_gb = 0.0
            current_size_in_gb += size_in_gb
            current_group.append(module)
        if len(current_group) > 0:
            module_prefetch_groups.append(current_group)
        return module_prefetch_groups

    def _prepare_load_order(
        self, load_chunks: List[List[torch.nn.Module]]
    ) -> Dict[int, List[ModuleOffloading]]:
        wrapped_modules = []
        for _, module in self.model.named_modules():
            if isinstance(module, ModuleOffloading):
                wrapped_modules.append(module)

        module_id_to_wrapped = {id(w.module): w for w in wrapped_modules}
        result = {}
        for chunk, next_chunk in zip(load_chunks, load_chunks[1:]):
            result[id(chunk[0])] = [module_id_to_wrapped[id(m)] for m in next_chunk]

        return result


def _schedule_prefetch(module: ModuleOffloading, forward: bool):
    module_to_next_chunk = (
        module.ctx.module_to_fetching_list_forward
        if forward
        else module.ctx.module_to_fetching_list_backward
    )
    next_modules_to_load = module_to_next_chunk[id(module.module)]
    if next_modules_to_load is not None:
        for module_to_load in next_modules_to_load:
            module_to_load.schedule_load()


def _wrap_all_submodules(
    module: torch.nn.Module,
    wrapper_module_cls: Callable[[torch.nn.Module], torch.nn.Module],
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
