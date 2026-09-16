# Contributing

Contributions should be focused, tested, and free of machine-specific data. Do not commit model
weights, credentials, request contents, absolute home-directory paths, benchmark machine details,
or generated runtime libraries.

Before opening a change, run the formatting, build, lint, test, notice, and privacy commands
documented in the README. New behavior requires tests. Public contracts require an architecture
decision record when they introduce a lasting constraint. Regenerate `THIRD_PARTY_NOTICES.md` after
changing the locked dependency graph.

Dependencies must be necessary, actively maintained, and compatible with the repository's
dual-license policy. Document exceptions before adding them.

Do not publish benchmark results with host details, replace tagged release assets manually, or mark
a model semantically verified without exact-artifact golden-vector evidence and an independent
review.
