from dataclasses import dataclass
from model.transformer import TransformerLM
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
    Runs warmup steps and then benchmarks model forward (and optionally backward) passes
    on random data.
    """

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
    for _ in range(config.warmup_steps):

        model(input)

    for _ in range(config.benchmark_steps):

        model(input)

if __name__ == "__main__":

    config = BenchmarkConfig(
        warmup_steps=5,
        benchmark_steps=100,
        forward_only=True,
        batch_size=4,
        d_model=256,
        vocab_size=10_000,
        context_length=256,
        num_layers=12,
        num_heads=12,
        d_ff=1344,
        rope_theta=10_000,
        device="cpu",
        dtype=torch.float32
    )

    benchmark(config=config)