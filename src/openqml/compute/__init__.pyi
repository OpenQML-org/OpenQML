"""Static public exports; runtime loading stays lazy."""

from . import kernel as kernel
from .kernel import ComputeKernel as ComputeKernel
from .kernel import ComputePolicy as ComputePolicy
from .kernel import KernelOps as KernelOps
from .kernel import RealOps as RealOps
from .kernel import compute_context as compute_context
from .kernel import compute_operation as compute_operation
from .kernel import configure_compute as configure_compute
from .kernel import floating_array as floating_array
from .kernel import get_compute_kernel as get_compute_kernel
from .kernel import memory_budget as memory_budget
