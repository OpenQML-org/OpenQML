# Governance

OpenQML is maintained through its GitHub repository. Existing repository
maintainers and administrators retain technical responsibility for reviewing,
merging and releasing changes. This document grants no account permissions.

## Decisions

Routine fixes and maintenance are handled in pull requests. Maintainers review
the behavior, applicable CI, numerical checks, compatibility and documentation.
Changes spanning multiple workflows describe their shared execution path and
integration checks in the pull request. Disagreements and their resolution are
recorded there; unresolved concerns can be raised in a repository issue.

New public APIs, breaking changes, dependencies and substantial algorithmic
changes need a documented rationale, limitations and migration impact before
merging. Public API removals require an announced deprecation or an explicitly
documented breaking release. Unused internal implementations can be removed
with their callers and obsolete documentation.

## Maintainers and access

Repository administrators manage maintainer access using GitHub settings.
Additions and removals should record the scope and responsible administrator
in a project decision without exposing credentials. Contribution volume alone
does not grant merge, release or security access.

Release credentials stay outside the repository. CI actions use immutable
commit references and minimum job permissions. Publishing uses the repository's
existing PyPI trusted publisher after its test/build gates. Documentation must
distinguish measured results from capabilities that have not been validated on
the claimed hardware.

Use repository issues or discussions to contact maintainers about project
decisions. Conduct concerns follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and vulnerability reports follow [SECURITY.md](SECURITY.md).
