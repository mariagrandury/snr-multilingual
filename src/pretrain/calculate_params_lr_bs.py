"""
Calculate the number of total and non-embedding parameters for a model, as well as the learning rate and batch size to train the model.

Credits to A. Hägele.
"""

import math


def parameter_count(
    vocab_size,  # vocabulary size
    n_layers,  # number of layers
    d_model,  # hidden size
    num_heads,  # number of attention heads
    num_kv_heads,  # number of key/value heads
    ffw_size,  # feed-forward size
    n_experts=1,  # number of experts
    swiglu_or_geglu=False,  # activation function
    tied_weights=False,  # whether input/output embeddings share weights
    verbose=True,
):
    """
    Compute the number of total and non-embedding parameters for a model.
    """
    head_dim = d_model / num_heads
    mul_factor_ffn = 3 if swiglu_or_geglu else 2
    attn = 2 * d_model * num_heads * head_dim + 2 * d_model * num_kv_heads * head_dim
    tied_factor = 1 if tied_weights else 2
    emb = tied_factor * vocab_size * d_model
    layers = n_layers * (attn + mul_factor_ffn * n_experts * d_model * ffw_size)
    if verbose:
        print(
            f"Model size: n_layers={n_layers}, d_model={d_model}, ratio={d_model/n_layers:.2f}\n"
        )
        print("Parameter count:")
        print(f"  Embedding params: \t {emb / 1e9:.3f}B")
        print(f"  Layer params: \t\t {layers / 1e9:.3f}B")
        print(f"  Total params: \t\t {(emb + layers) / 1e9:.3f}B\n")
    return emb + layers, layers


def approximate_real_bs(theoretical_bs, micro_batch_size, seq_len, gpu_group_size=4):
    """
    Approximate the real batch size in tokens and number of micro-batches.
    """
    # Batch size in tokens per micro-batch
    mbs_tokens = micro_batch_size * seq_len
    # Number of micro-batches (raw)
    raw_num_mbs = theoretical_bs / mbs_tokens
    # Round to nearest multiple of gpu_group_size
    num_mbs = round(raw_num_mbs / gpu_group_size) * gpu_group_size
    # Ensure we have at least one group if the theoretical BS is very small
    num_mbs = max(num_mbs, gpu_group_size)
    # Real batch size in tokens
    return num_mbs * mbs_tokens, num_mbs


def get_learning_rate(C):
    return 0.3118 * C**-0.1250


def get_batch_size(C):
    return 0.2920 * C**+0.3271


def round_to_R(data_size, batch_size, R=2000):
    # Calculate the raw division result
    result = data_size / batch_size

    # Round up to the nearest multiple of R
    rounded_result = math.ceil(result / R) * R

    return int(rounded_result)


def set_lr_and_bs(
    num_parameters,
    desired_tokens,
    n_layers,
    d_model,
    seq_len,
    micro_batch_size,
    verbose=True,
):
    """
    Set the learning rate and batch size for a model.

    The number of total training FLOPs is estimated in two ways:
    - the architecture-agnostic Chinchilla formula C=6ND (https://arxiv.org/abs/2203.15556), used for the LR and BS calculations
    - the architecture-aware DeepSeek formula (https://arxiv.org/abs/2401.02954), printed for comparison
    """
    C = 6 * num_parameters * desired_tokens
    C_ = (
        72 * n_layers * d_model**2 + 12 * n_layers * d_model * seq_len
    ) * desired_tokens  # DeepSeek FLOP formula
    lr = get_learning_rate(C)
    theoretical_bs = get_batch_size(C)

    real_bs, num_mbs = approximate_real_bs(theoretical_bs, micro_batch_size, seq_len)
    its = round_to_R(desired_tokens, real_bs)

    if verbose:
        print("LR and BS calculations:")
        print(
            f"  6ND:  \t C={C:.3e}  lr={lr:.6f}  theoretical_bs={theoretical_bs:.0f}\n"
            f"        \t real_bs={real_bs}  num_mbs={num_mbs}  iters={its}"
        )
        print(
            f"  DeepSeek:\t C={C_:.3e}  lr={get_learning_rate(C_):.6f}  bs={get_batch_size(C_):.0f}\n"
        )
    return lr, real_bs, its, num_mbs
