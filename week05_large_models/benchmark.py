from typing import Tuple

import torch
import transformers
from detach import Detach
from offloading import Offloading


def run_benchmark():
    print("Loading model...")
    model, tokenizer = get_model()
    print("Model loaded.")

    print("Running forward/backward benchmark...")
    forward_backward_benchmark(model, tokenizer)

    print("Running generation benchmark...")
    generation_benchmark(model, tokenizer)


def get_model() -> Tuple[torch.nn.Module, transformers.AutoTokenizer]:
    model_name = "facebook/opt-iml-1.3b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    for name, module in model.named_modules():
        requires_grad = name.startswith("model.decoder.layers.23") or name.startswith(
            "lm_head"
        )
        for _, p in module.named_parameters(recurse=False):
            p.requires_grad = requires_grad
    model.model.decoder.layers[23] = Detach(model.model.decoder.layers[23])

    inp = torch.randint(0, 10, (2, 3), device="cpu")
    model_witH_offloading = Offloading(
        model,
        sample_input=inp,
        cuda_device_idx=0,
        max_size_in_gb=0.3,
        prefetch=True,
    )
    return model_witH_offloading, tokenizer


def forward_backward_benchmark(model, tokenizer):
    inp = torch.randint(0, 10, (128, 1024), device="cuda:0")

    print("running forward")
    outp = model(inp)

    print("running backward")
    losss = outp.logits.sum()
    losss.backward()


def generation_benchmark(model, tokenizer):
    batch = tokenizer(["A cat sat", "import numpy"], return_tensors="pt")
    batch = {name: tensor.cuda() for name, tensor in batch.items()}
    print("running generation")
    # generated_ids = model.generate(**batch, max_length=32)
    print("done generation")
    print("Sample A:", tokenizer.decode(generated_ids[0]))
    print("Sample B:", tokenizer.decode(generated_ids[1]))


if __name__ == "__main__":
    run_benchmark()
