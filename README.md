# jacta-manipulation
[![build](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml/badge.svg)](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/bdaiinstitute/jacta-manipulation/graph/badge.svg?token=SH5Y2J032M)](https://codecov.io/gh/bdaiinstitute/jacta-manipulation)
[![Static Badge](https://img.shields.io/badge/documentation-latest-8A2BE2)](https://upgraded-disco-qzn7k5e.pages.github.io)

> **Disclaimer**  
> This code is released in conjunction with the publication of _"Jacta: A Versatile<br>
> Planner for Learning Dexterous and Whole-body Manipulation."_ It is provided<br>
> as a research prototype and is not production-quality software. Please note<br>
> that the code may contain missing features and potential bugs. As part of this<br>
> release, the RAI Institute does not offer maintenance or support for the software.

### Install package
```
git clone https://github.com/bdaiinstitute/jacta-manipulation.git
cd jacta-manipulation
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run tests locally
In the virtual environment:
```
pip install -e .[dev]
python -m pytest
```
you might have to
```
unset PYTHONPATH
```
