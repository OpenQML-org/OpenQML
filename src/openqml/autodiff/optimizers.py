"""Optional optimizer factories without importing a framework during discovery."""


def hybrid_qnm(model, **options):
    """Build AdamW classical groups and distance-aware QNM quantum groups."""
    try:
        from ..qml.interfaces._torch_optimizer import HybridQNM
    except ImportError as exc:
        from ..errors import MissingDependencyError

        raise MissingDependencyError(
            'Cannot load PyTorch. Run: python -m pip install "openqml[torch]".'
        ) from exc

    return HybridQNM(model, **options)
