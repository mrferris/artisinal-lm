import argparse
from dataclasses import dataclass
from lm.training.loss.cross_entropy import cross_entropy
from lm.model.transformer import TransformerLM
from timeit import timeit
import torch

@dataclass
class BenchmarkConfig:
    """
    Configures benchmarking and the model parameters to be benchmarked.
    """
    warmup_steps: int
    benchmark_steps: int
    forward_only: bool

    batch_size: int
    d_model: int
    vocab_size: int
    context_length: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: int
    device: torch.device
    dtype: torch.dtype
    compile: bool


def benchmark(config: BenchmarkConfig):
    """
    Runs warmup steps and then times model forward (and optionally backward) passes
    on random data.
    """
    print("============================================")
    print("Running benchmarks: ")
    print(f"warmup_steps={config.warmup_steps}, benchmark_steps={config.benchmark_steps}, forward_only={config.forward_only}")
    print("Model hyperparameters:")
    print(
        f"  d_model={config.d_model}, vocab_size={config.vocab_size}, context_length={config.context_length}\n"
        f"  num_layers={config.num_layers}, num_heads={config.num_heads}, d_ff={config.d_ff}, rope_theta={config.rope_theta}\n"
        f"  device={config.device}, dtype={config.dtype}, batch_size={config.batch_size},"        
    )
    model = TransformerLM(
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=config.dtype,
    )
    model.to(config.device)
    print(f"Non-embedding param count: {model.param_count[1]}")

    input = torch.rand(config.batch_size, config.context_length, config.d_model)
    desired_output = torch.rand(config.batch_size, config.context_length, config.d_model)
    
    # Warmup steps
    for _ in range(config.warmup_steps):

        output = model(input)        
        if config.forward_only:
            continue
        loss = cross_entropy(output, desired_output)
        loss.backward()        

    # Benchmarking
    for _ in range(config.benchmark_steps):

        output = model(input)
        if config.forward_only:
            continue
        loss = cross_entropy(output, desired_output)
        loss.backward()

if __name__ == "__main__":

    arguments = argparse.ArgumentParser(description="Benchmark LLM")
    arguments.add_argument("--batch-size", type=int, default=4, help="Number of batches per training step")
    arguments.add_argument("--context-length", type=int, default=256, help="length of model's context length")
    arguments.add_argument("--d-model", type=int, default=512, help="Dimension of model's embeddings")
    arguments.add_argument("--vocab-size", type=int, default=10_000, help="Number of tokens in the model's vocab")
    arguments.add_argument("--num-heads", type=int, default=16, help="Heads per attention mechanism in the model")
    arguments.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers in the model")
    arguments.add_argument("--d-ff", type=int, default=1344, help="Dimension of the feedforward networks in the model")
    arguments.add_argument("--rope-theta", type=int, default=10000, help="Constant used in RoPE rotation calculations")
    arguments.add_argument("--warmup-steps", type=int, default=5, help="Number of steps before beginning benchamrk measurements")
    arguments.add_argument("--benchmark-steps", type=int, default=10, help="Number of steps to benchmark")
    arguments.add_argument("--forward-only", type=bool, default=False, help="Run benchmarking on forward passes only, not backward")
    arguments.add_argument("--device", type=str, default="cuda", help="Device on which to run benchmarks")
    args = arguments.parse_args()

    config = BenchmarkConfig(
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        forward_only=args.forward_only,
        batch_size=args.batch_size,
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=args.dtype
    )

    benchmark(config=config)