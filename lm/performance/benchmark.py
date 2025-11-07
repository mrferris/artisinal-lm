import argparse
import statistics
import timeit
from dataclasses import dataclass

import torch

from lm.model.transformer import TransformerLM
from lm.performance.utils import synchronize_accelerator
from lm.training.loss.cross_entropy import cross_entropy


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
    compile: bool


def benchmark(config: BenchmarkConfig):
    """
    Runs warmup steps and then times model forward (and optionally backward) passes
    on random data.
    """
    print("=======================================================")
    print("Running benchmarks: ")
    print(f"warmup_steps={config.warmup_steps}, benchmark_steps={config.benchmark_steps}, forward_only={config.forward_only}")
    print("Model hyperparameters:")
    print(
        f"  d_model={config.d_model}, vocab_size={config.vocab_size}, context_length={config.context_length}\n"
        f"  num_layers={config.num_layers}, num_heads={config.num_heads}, d_ff={config.d_ff}, rope_theta={config.rope_theta}\n",
    )
    print(f"device={config.device}, compile={config.compile}, batch_size={config.batch_size}")
    model = TransformerLM(
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
    )
    param_count = model.param_count()[1]
    print(f"Non-embedding param count: {param_count:,}")
    print("=======================================================")

    input = torch.randint(low=0, high=config.vocab_size - 1, size=(config.batch_size, config.context_length))
    desired_output = torch.randint(low=0, high=config.vocab_size - 1, size=(config.batch_size, config.context_length))

    model.to(config.device)
    input = input.to(config.device)
    desired_output = desired_output.to(config.device)

    # Warmup steps
    for _ in range(config.warmup_steps):
        model_step(model, input, desired_output, config.forward_only)

    # Benchmarking
    times = timeit.repeat(lambda: model_step(model, input, desired_output, config.forward_only), number=1, repeat=config.benchmark_steps)

    mean = statistics.mean(times)
    stdev = statistics.stdev(times)

    print(f"Times: {times}")
    print(f"Mean: {mean} seconds")
    print(f"Std dev: {stdev} seconds")


def model_step(model, input, desired_output, forward_only):
    output = model(input)
    if not forward_only:
        output = output.to(config.device)
        loss = cross_entropy(output, desired_output)
        loss.backward()

    synchronize_accelerator(config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of batches per training step")
    parser.add_argument("--context-length", type=int, default=256, help="length of model's context length")
    parser.add_argument("--d-model", type=int, default=512, help="Dimension of model's embeddings")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Number of tokens in the model's vocab")
    parser.add_argument("--num-heads", type=int, default=16, help="Heads per attention mechanism in the model")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers in the model")
    parser.add_argument("--d-ff", type=int, default=1344, help="Dimension of the feedforward networks in the model")
    parser.add_argument("--rope-theta", type=int, default=10_000, help="Constant used in RoPE rotation calculations")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Steps before beginning benchamrk measurements")
    parser.add_argument("--benchmark-steps", type=int, default=10, help="Number of steps to benchmark")
    parser.add_argument("--forward-only", dest="forward_only", action="store_true", help="Benchmark forward passes only")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run benchmarks")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile")
    parser.set_defaults(compile=True, forward_only=False)
    args = parser.parse_args()

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
        compile=args.compile,
    )

    benchmark(config=config)
