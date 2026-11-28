"""Phase 2 - fine-tune GigaAM-Multilingual on the Uzbek data built by prepare_data.py.

Wraps the official GigaAM train_utils/train.py (third_party/GigaAM, see setup_gigaam_train.sh) with:
  * a fast FLAC reader (the official loader starts one ffmpeg process per clip for non-WAV files)
  * defaults that fit one 24 GB L4 and our small disk (bf16, activation checkpointing,
    weights-only checkpoints, only the best 2 kept)

    python train_gigaam.py --trial                 # ~100 steps: checks memory + measures speed
    python train_gigaam.py --max-steps 6000        # real run (run it inside tmux!)
    python train_gigaam.py --max-steps 6000 --resume checkpoints/models/<exp>/<file>.ckpt
    python train_gigaam.py --trial -- --freq_masks 0   # anything after "--" goes to train.py as-is

Checkpoints: checkpoints/models/<experiment>/gigaam-multilingual_large_ctc-...-val_wer=0.xxxx.ckpt
Evaluate one: python asr_baseline.py --model checkpoints/models/<experiment>/<file>.ckpt
TensorBoard:  tensorboard --logdir checkpoints  (then ssh -L 6006:localhost:6006 kronk)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GIGAAM_DIR = os.path.join(HERE, "third_party", "GigaAM")


def patch_fast_audio_loading():
    """Read 16 kHz mono files (our FLAC) with soundfile directly instead of spawning ffmpeg."""
    import soundfile as sf
    import torch
    from gigaam.utils import AudioDataset

    original = AudioDataset._load_audio

    def fast_load(item):
        if isinstance(item, str):
            try:
                with sf.SoundFile(item) as f:
                    if f.samplerate == 16000 and f.channels == 1:
                        return torch.from_numpy(f.read(dtype="float32"))
            except RuntimeError:
                pass
        return original(item)

    AudioDataset._load_audio = staticmethod(fast_load)


def patch_validation_mode():
    """Validate under no_grad instead of torch.inference_mode.

    GigaAM's rotary position encoding keeps a buffer that grows when a longer clip arrives. If it
    grows during validation (inference mode), the next training step crashes with
    "Inference tensors cannot be saved for backward". Lightning's inference_mode=False avoids it.
    """
    import pytorch_lightning as pl

    original = pl.Trainer

    class Trainer(original):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("inference_mode", False)
            super().__init__(*args, **kwargs)

    pl.Trainer = Trainer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/gigaam_uz", help="folder with train.tsv / val.tsv")
    p.add_argument("--train", help="train manifest (default: <data>/train.tsv)")
    p.add_argument("--val", help="validation manifest (default: <data>/val.tsv)")
    p.add_argument("--model", default="multilingual_large_ctc",
                   help="GigaAM model name (multilingual_large_ctc = 600M, multilingual_ctc = 220M)")
    p.add_argument("--trial", action="store_true", help="short run: 100 steps, validate at 50")
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--val-every", type=int, default=1000, help="validate + checkpoint every N steps")
    p.add_argument("--val-batches", type=int, default=60, help="validation batches per check (speed)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=4, help="effective batch = batch-size x accumulate")
    p.add_argument("--lr", type=float, default=2e-5, help="low LR: adapt to Uzbek without forgetting")
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--name", default=None, help="experiment name (default: built from settings)")
    p.add_argument("--resume", default=None, help="continue from a .ckpt")
    p.add_argument("--cpu", action="store_true", help="debug on CPU (tiny runs only)")
    args, passthrough = p.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    train = args.train or os.path.join(args.data, "train.tsv")
    val = args.val or os.path.join(args.data, "val.tsv")
    for path in (train, val):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} - run prepare_data.py first (or pass --train/--val).")
    if not os.path.isdir(os.path.join(GIGAAM_DIR, "train_utils")):
        raise SystemExit("third_party/GigaAM not found - run: bash setup_gigaam_train.sh")

    if args.trial:
        args.max_steps, args.val_every, args.val_batches = 100, 50, 20

    # Python 3.14 changed the Linux default for new processes from "fork" to "forkserver", which has
    # to pickle the dataset for each DataLoader worker - and GigaAM's dataset holds lambdas that
    # can't be pickled ("Can't pickle local object AudioDataset._make_encoder.<locals>.<lambda>").
    # "fork" (the default before 3.14) copies the process instead, so nothing needs pickling.
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    sys.path.insert(0, os.path.join(GIGAAM_DIR, "train_utils"))
    patch_fast_audio_loading()
    patch_validation_mode()
    import train as gigaam_train  # official script

    argv = [
        "train.py",
        "--model_name", args.model,
        "--train_manifest", train,
        "--val_manifest", val,
        "--raw_text",
        "--max_steps", str(args.max_steps),
        "--val_check_steps", str(args.val_every),
        "--val_first_batches", str(args.val_batches),
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.batch_size * 2),
        "--accumulate_grad_batches", str(args.accumulate),
        "--lr", str(args.lr),
        "--max_duration", str(args.max_duration),
        "--precision", "32" if args.cpu else args.precision,
        "--num_workers", str(args.workers),
        "--output_dir", args.out,
        "--save_top_k", "2",
        "--save_weights_only",  # 2.4 GB per checkpoint instead of ~7 GB with optimizer state
    ]
    if not args.cpu:
        argv.append("--activation_checkpointing")
    else:
        argv += ["--accelerator", "cpu"]
    if args.name:
        argv += ["--exp_name", args.name]
    if args.resume:
        argv += ["--resume_from_checkpoint", args.resume]
    argv += passthrough

    print("Running official GigaAM train.py with:\n  " + " ".join(argv[1:]), flush=True)
    sys.argv = argv
    gigaam_train.main()


if __name__ == "__main__":
    main()
