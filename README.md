## jacta-manipulation
[![build](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml/badge.svg)](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/bdaiinstitute/jacta-manipulation/graph/badge.svg?token=SH5Y2J032M)](https://codecov.io/gh/bdaiinstitute/jacta-manipulation)
[![Static Badge](https://img.shields.io/badge/documentation-latest-8A2BE2)](https://upgraded-disco-qzn7k5e.pages.github.io)


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
pip install pytest
python -m pytest
```
you might have to
```
unset PYTHONPATH
```


# TODO
break deps on common and visualizers