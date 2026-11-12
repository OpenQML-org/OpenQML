# Security policy

Security fixes target the current public PyPI release. Users of older releases
should upgrade; any backport is announced explicitly by the maintainers.

## Reporting

Use **Security → Report a vulnerability** on this repository when GitHub private
vulnerability reporting is available. Include the affected version, impact,
necessary input and relevant environment details. Remove tokens, credentials,
personal data and confidential workloads.

If the private reporting option is unavailable, request a private contact
channel through a repository issue without publishing vulnerability details.
This policy does not claim that private reporting is enabled or promise a
response deadline. Maintainers coordinate a fix and disclosure through GitHub
security advisories when appropriate.

## Scope

Reports may concern unsafe parsing/loading, unintended code execution, resource
limit bypasses, credential exposure or untrusted extension boundaries. Numerical
accuracy and performance defects normally belong in ordinary issues unless
their impact creates a security concern.

Registered adapters and backends execute Python code supplied by the caller;
they are not a sandbox. Do not execute untrusted Python or load untrusted pickle
files. Keep QPU credentials in the provider's supported configuration, and
exclude them from circuit metadata, checkpoints and bug reports.
