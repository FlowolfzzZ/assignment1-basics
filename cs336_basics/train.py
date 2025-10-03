import torch
import argparse
from cs336_basics.Transformer import Transformer
from cs336_basics.Optimizer import AdamW
from cs336_basics.functions import cross_entropy, learning_rate_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--vocab_size", type=int, required=False, default=10000, help='Vocabulary size.')
    parser.add_argument("--context_length", type=int, required=False, default=256, help='Maximum context length.')
    parser.add_argument("--d_model", type=int, required=False, default=512, help='Model dimension.')
    parser.add_argument("--num_layers", type=int, required=False, default=4, help='Transformer layers count.')
    parser.add_argument("--num_heads", type=int, required=False, default=16, help='Attention heads count.')
    parser.add_argument("--d_ff", type=int, required=False, default=1344, help='FFN dimension.')
    parser.add_argument("--rope_theta", type=float, required=False, default=10000, help='RoPE theta parameter.')
    parser.add_argument("--batch_size", type=int, required=False, default=512, help='Batch size.')

    # parser.add_argument("--vocab_path", type=str, required=False, default="", help='Vocabulary input path.')
    # parser.add_argument("--merge_path", type=str, required=True, help='Vocabulary merge operations\' input path.')
    # parser.add_argument("--special_tokens", nargs='+', type=str, required=False, default=None, help='Checkpoint output path.')

    parser.add_argument("--max_learning_rate", type=float, required=False, default=3e-3, help='Learning rate.')
    parser.add_argument("--min_learning_rate", type=float, required=False, default=3e-4, help='Learning rate.')
    parser.add_argument("--weight_decay", type=float, required=False, default=0.01, help='Weight decay.')
    parser.add_argument("--betas", nargs=2, type=float, required=False, default=[0.9, 0.999], help='AdamW betas.')
    parser.add_argument("--eps", type=float, required=False, default=1e-8, help='AdamW eps.')
    parser.add_argument("--grad_clip", type=float, required=False, default=1e-2, help='Gradient clipping max_l2_norm.')
    parser.add_argument("--warmup_steps", type=int, required=False, default=50, help='Warmup steps.') 
    parser.add_argument("--max_steps", type=int, required=False, default=2500, help='Maximum training steps.')

    parser.add_argument("--training_dataset", type=str, required=False, default="results/token_ids/tinystories-vocab/TinyStoriesV2-GPT4-train.npy",
                        help='Training dataset path.')
    parser.add_argument("--validation_dataset", type=str, required=False, default="results/token_ids/tinystories-vocab/TinyStoriesV2-GPT4-valid.npy",
                        help='Validation dataset path.')
    parser.add_argument("--validation_interval", type=int, required=False, default=100, help='Validation interval steps.')
    parser.add_argument("--validation_iters", type=int, required=False, default=100, help='Validation iterations.')

    parser.add_argument("--device", type=str, required=False, default="cuda:0", help='Device name.')
    parser.add_argument("--checkpoint_path", type=str, required=False, default=None, help='Checkpoint path to load.')
    parser.add_argument("--save_path", type=str, required=False, default="results/training_TinyStories", help='Checkpoint output path.')

    args = parser.parse_args()
    device = args.device
    print("*" * 20)
    print(f"Training arguments: {args}")
    print("*" * 20)

    model = Transformer(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta, device=device, dtype=torch.bfloat16)
    # model = torch.compile(model, backend="aot_eager")
    optimizer = AdamW(model.parameters(), lr=args.max_learning_rate, weight_decay=args.weight_decay, betas=tuple(args.betas), eps=args.eps)

    training_dataset = np.load(args.training_dataset, mmap_mode="r")
    validation_dataset = np.load(args.validation_dataset, mmap_mode="r")
    start_time = time.time()

    start_iter = 1
    if args.checkpoint_path is not None:
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)

    for t in range(start_iter, args.max_steps + 1):
        lr = learning_rate_schedule(t, args.max_learning_rate, args.min_learning_rate, args.warmup_steps, args.max_steps)
        optimizer.defaults["lr"] = lr
        x, y = get_batch(training_dataset, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if t % args.validation_interval == 0 or t == args.max_steps:
            model.eval()
            with torch.no_grad():
                loss = 0
                for _ in range(args.validation_iters):
                    x, y = get_batch(validation_dataset, args.batch_size, args.context_length, device)
                    logits = model(x)
                    loss += cross_entropy(logits, y)
                loss /= args.validation_iters
                end_time = time.time()
                print(f"Step: {t}. Validation loss: {loss.cpu().item()}. Learning rate: {lr}. Elapsed time: {end_time - start_time:.2f} seconds.")
                save_checkpoint(model, optimizer, t, f"{args.save_path}/checkpoint_{t}")
            model.train()

    end_time = time.time()
    print("*" * 20)
    print(f"Training finished!")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds.")
    print("*" * 20)
    save_checkpoint(model, optimizer, t, f"{args.save_path}/final_model")