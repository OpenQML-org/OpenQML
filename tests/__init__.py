import numpy as np


def as_numpy(value):
    """Snapshot a native test result on the host without retaining its graph."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
