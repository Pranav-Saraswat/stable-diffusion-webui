import torch

# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
has_mps = getattr(torch, 'has_mps', False)

cpu = torch.device("cpu")


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("mps") if has_mps else cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
