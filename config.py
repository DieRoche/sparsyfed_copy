import argparse
import random


def str2bool(v):
    """Convert a string representation of truth to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Averaging Experiments")
    parser.add_argument("--method", type=str, default="Sparsyfed")
    parser.add_argument("--n_client", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=0.5)
    parser.add_argument("--dirichlet", type=float, default=0.5)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--n_client_epoch", type=int, default=5)
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--seed", type=int, default=5)

    parser.add_argument("--sparisty", type=float, default=0.95)
    

    parser.add_argument("--device", type=str, default="cuda")

    args, _ = parser.parse_known_args()

    return args
