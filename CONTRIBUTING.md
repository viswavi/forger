# Contributing to Forger

Thanks for your interest in contributing to Forger!
We appreciate your support and welcome your
contributions. Here's a guide to help you get started:

## Developer Installation

Installing dependencies:
Run the following command to install all the dependencies after you have forked the repository.

```bash
pip install .[dev]
```

You should be using Python 3.12+ (our CI specifically tests Python 3.12 and 3.13).

If you're a developer, it's recommended to install
pre-commit hooks before starting your development
work. These hooks ensure code formatting, linting,
and type-checking. To install the pre-commit hooks,
run the following command:

```bash
pre-commit install
```

Additionally, it's essential to run tests to verify the
functionality of your code. Execute the following
command to run the tests:

```bash
pytest
```

## Op Reference

`docs/OPS.md` lists the signature of every public op and is generated from the code.
After adding an op or changing an op's signature or docstring, regenerate it:

```bash
python scripts/gen_op_docs.py
```

`tests/test_op_docs.py` fails until `docs/OPS.md` is up to date. A new exported class that
users construct and pass to ops goes in `ENTRY_POINT_CLASSES` in the script.

## Making a Release

If you have admin privileges for the repository,
you can create a new release of the forger
library. We utilize the
[hatchling](https://github.com/pypa/hatch) build
system, which simplifies the process of making
new releases.

To create a new release, follow these steps:

1. Create a new version tag on GitHub, adhering to
the [semantic versioning](https://semver.org/) guidelines.
2. Once the tag is created, the continuous integration
(CI) system will automatically build and publish the
new version to PyPI.

By following these steps, you can effectively make
new releases of the library and contribute to its
ongoing development.
