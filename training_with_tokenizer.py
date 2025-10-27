"""
Training script for discrete diffusion model with HuggingFace tokenizer
Optimized for Vietnamese language using PhoGPT tokenizer
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from model import (
    DiffusionTransformer,
    DiffusionConfig,
    encode_text,
    decode_tokens,
    MaskedDiffusionSchedule,
)
from sample import get_random_context


def get_data_loader_with_tokenizer(data_path, tokenizer, batch_size, seq_len, device):
    """
    Create data loader using HuggingFace tokenizer

    Args:
        data_path: Path to text file
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to load data on

    Returns:
        data_generator: Generator that yields batches
        tokens: Full dataset tokens (for context sampling)
    """
    # Read the text file
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded text: {len(text):,} characters")

    # Convert to tokens using tokenizer
    tokens = encode_text(text, tokenizer)
    print(f"Encoded to: {len(tokens):,} tokens")

    # Create batches
    num_batches = len(tokens) // (batch_size * seq_len)
    tokens = tokens[: num_batches * batch_size * seq_len]
    tokens = tokens.view(batch_size, -1)

    print(f"Dataset shape: {tokens.shape}")
    print(f"Number of sequences: {tokens.size(1) // seq_len}")

    # Generator function
    def data_generator():
        while True:
            for i in range(0, tokens.size(1) - seq_len, seq_len):
                batch = tokens[:, i : i + seq_len].to(device)
                yield batch

    return data_generator(), tokens


def train_step(model, x_0, mask_schedule, optimizer):
    """
    Single training step

    Args:
        model: DiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        mask_schedule: Mask schedule object
        optimizer: Optimizer

    Returns:
        loss: Training loss
    """
    B, _ = x_0.shape
    device = x_0.device

    # Sample random timesteps
    t = torch.randint(0, mask_schedule.num_timesteps, (B,), device=device)

    # Add mask to get x_t
    x_t = mask_schedule.add_masks(x_0, t)

    # Forward pass: predict the original tokens
    logits = model(x_t, t)  # (B, T, vocab_size)

    # Compute loss only on masked positions
    mask = x_t == mask_schedule.mask_token_id  # (B, T)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="none"
    )
    loss = (
        loss.view(B, -1) * mask
    ).sum() / mask.sum()  # Average over masked positions only

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    model,
    data_loader,
    mask_schedule,
    optimizer,
    tokenizer,
    num_steps=10000,
    sample_interval=500,
    save_interval=2000,
    dataset_tokens=None,
):
    """
    Main training loop
    """
    model.train()

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Get batch
        x_0 = next(data_loader)

        # Training step
        loss = train_step(model, x_0, mask_schedule, optimizer)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Sample generation
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get random context if context_len > 0
                context_tokens = None
                if model.config.context_len > 0 and dataset_tokens is not None:
                    context_tokens = get_random_context(
                        dataset_tokens, model.config.context_len, batch_size=1
                    )

                samples = model.sample(
                    batch_size=1,
                    seq_len=model.config.sequence_len,
                    mask_schedule=mask_schedule,
                    num_steps=None,  # Use all timesteps
                    temperature=1.0,
                    device=model.get_device(),
                    context_tokens=context_tokens,
                )

                # Decode samples to text using tokenizer
                text = decode_tokens(samples[0], tokenizer)
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text)
                tqdm.write("--- End sample ---\n")
            model.train()

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            checkpoint_path = f"weights/checkpoint_step_{step + 1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config,
                    "tokenizer_name": tokenizer.name_or_path,
                    "loss": loss,
                },
                checkpoint_path,
            )
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")


def main():
    # Hyperparameters
    batch_size = 64
    max_iters = 20000
    eval_interval = 500
    save_interval = 2000
    learning_rate = 3e-4

    # Tokenizer
    print("Loading Vietnamese tokenizer...")
    tokenizer_name = "vinai/PhoGPT-4B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Get vocab size and mask token
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else 0

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Mask token ID: {mask_token_id}")

    # Configuration
    config = DiffusionConfig(
        sequence_len=256,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        n_layer=6,
        n_head=6,
        n_embd=384,
        diffusion_steps=128,
        context_len=64,
    )

    print(f"\nModel configuration:")
    print(f"  Sequence length: {config.sequence_len}")
    print(f"  Diffusion steps: {config.diffusion_steps}")
    print(f"  Context length: {config.context_len}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # Model
    model = DiffusionTransformer(config).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Masked diffusion schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps,
        mask_token_id=config.mask_token_id,
        context_len=config.context_len,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Data path (modify this to your data file)
    data_path = "data/vietnamese_train.txt"

    print(f"\nLoading data from: {data_path}")

    # Data loader
    data_loader, dataset_tokens = get_data_loader_with_tokenizer(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=config.sequence_len,
        device=device,
    )

    # Train
    print("\nStarting training...\n")
    train(
        model=model,
        data_loader=data_loader,
        mask_schedule=mask_schedule,
        optimizer=optimizer,
        tokenizer=tokenizer,
        num_steps=max_iters,
        sample_interval=eval_interval,
        save_interval=save_interval,
        dataset_tokens=dataset_tokens,
    )

    # Save final model
    import os
    os.makedirs("weights", exist_ok=True)

    final_path = "weights/vietnamese_diffusion_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size,
        },
        final_path,
    )
    print(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    main()
