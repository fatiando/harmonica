# Contributing Guidelines

:tada: **First off, thank you for considering contributing to our project!** :tada:

This is a community-driven project, so it's people like you that make it useful
and successful.
These are some of the many ways to contribute:

* :bug: Submitting bug reports and feature requests
* :memo: Writing tutorials or examples
* :mag: Fixing typos and improving the documentation
* :bulb: Writing code for everyone to use

**Please refer to our [organization-wide guidelines][contrib]**
for general instructions on how to contribute to Fatiando projects.

## Ground Rules

The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**.
Everyone must abide by our [Code of Conduct][coc] and we encourage all to read
it carefully.

## Authorship and credit

We strive to adequately reward and credit all those who contribute to our
project in any way.
This can vary from an acknowledgment in the release notes to authorship in
scientific publications.
**Please refer to our [Authorship Guidelines][authorship]** for more
information.

## For maintainers

You'll find more information about project maintenance (releases, reviews, etc)
in our organization-wide [Maintainers Guide][maintenance].

## About coverage analysis

Harmonica relies on numba for performance within some functions, which can make
it a bit difficult to measure test coverage since the functions being
executed are the jit-compiled versions of the functions in the code.
Fortunately, numba now supports measuring coverage if the environment variable
`NUMBA_JIT_COVERAGE=1` is set. However, it still does not support branch
analysis and will report any `for` or `if` as missing coverage for a branch,
even if tests explicitly for that are added. To get around this, we have to
disable branch analysis for every single `for` and `if` statement in our code by
putting `# pragma: no branch` after the statement.

[coc]: https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md
[contrib]: https://github.com/fatiando/community/blob/main/CONTRIBUTING.md
[maintenance]: https://github.com/fatiando/community/blob/main/MAINTENANCE.md
[authorship]: https://github.com/fatiando/community/blob/main/AUTHORSHIP.md
