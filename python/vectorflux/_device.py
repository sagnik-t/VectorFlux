"""
vectorflux/_device.py — Default device management.

Call vf.set_default_device('cuda') once before building the graph to place
all Variable weights and feed_dict tensors on the GPU automatically.
"""

_default_device: str = 'cpu'


def set_default_device(device: str) -> None:
    """
    Set the default device for the framework.

    Parameters
    ----------
    device : 'cpu' or 'cuda'

    Effects
    -------
    - New Variable weights are placed on this device.
    - Sessions created after this call auto-move feed_dict tensors to this device.
    - Optimizer updates preserve the variable's current device automatically.

    Call this once before building the graph, e.g.:
        vf.set_default_device('cuda')
        X_ph, Y_ph, logits, train_op = build_model()
        sess = vf.Session()
    """
    global _default_device
    if device not in ('cpu', 'cuda'):
        raise ValueError(
            f"set_default_device: unknown device '{device}'; "
            "expected 'cpu' or 'cuda'")
    _default_device = device


def get_default_device() -> str:
    """Return the current default device ('cpu' or 'cuda')."""
    return _default_device
