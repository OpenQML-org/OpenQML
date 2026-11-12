# Contributing

OpenQML develops three workflows: quantum–classical hybrid AI, quantitative
research, and cross-framework circuit programming. Proposals should explain
which workflow benefits and how the existing compute services are reused.

Use a focused pull request. Describe the changed behavior, numerical contract,
resource limits and relevant validation. Keep public documentation consistent
with the implementation. Avoid additional runtime dependencies when existing
NumPy, Torch or JAX primitives provide the required operation.

## Development

Install the core and development tools with `python -m pip install -e '.[dev]'`.
Install the integrations needed for the change; CI also exercises the complete
`.[all,dev]` installation. Run relevant tests, `ruff check .` and
`ruff format --check .`. Build with `python -m build` and check distributions
with `python -m twine check dist/*` when packaging changes.

Add a focused regression test for a corrected numerical, execution or API
contract. Keep standalone demonstrations, datasets and performance harnesses
out of the package. Performance statements must identify the workload,
precision, device and measured boundary. CPU checks do not establish GPU or
QPU throughput.

## Compatibility and release

Preserve supported Python/framework versions and documented public imports.
Validate public types before backend dispatch or large allocations. Invalid
inputs must raise an appropriate OpenQML exception rather than leaking an
incidental attribute or container error. Preserve autodiff and device placement.

Releases use the existing test/build workflow and PyPI trusted publishing.
The version in `pyproject.toml` and `openqml.__version__` must agree. Maintainers
verify the installed distribution against the merged source. Security reports
follow [SECURITY.md](SECURITY.md); project decisions follow
[GOVERNANCE.md](GOVERNANCE.md).
