# jacta-manipulation
[![build](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml/badge.svg)](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/bdaiinstitute/jacta-manipulation/graph/badge.svg?token=7A0OJ37JRF)](https://codecov.io/gh/bdaiinstitute/jacta-manipulation)
[![docs](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/docs.yml/badge.svg)](https://github.com/bdaiinstitute/jacta-manipulation/actions/workflows/docs.yml)
[![Static Badge](https://img.shields.io/badge/documentation-latest-8A2BE2)](https://upgraded-disco-qzn7k5e.pages.github.io)


> **Disclaimer**
> This code is released in conjunction with the publication of _"Jacta: A Versatile<br>
> Planner for Learning Dexterous and Whole-body Manipulation."_ It is provided<br>
> as a research prototype and is not production-quality software. Please note<br>
> that the code may contain missing features and potential bugs. As part of this<br>
> release, the RAI Institute does not offer maintenance or support for the software.

## Jacta: A Versatile Planner for Learning Dexterous and Whole-body Manipulation
<div align="center">

[![Static Badge](https://img.shields.io/badge/ArXiv-8C48FC?style=for-the-badge)](https://arxiv.org/pdf/2408.01258)
[![Static Badge](https://img.shields.io/badge/Project_Page-8C48FC?style=for-the-badge)](https://jacta-manipulation.github.io/)
[![Static Badge](https://img.shields.io/badge/RAI_Institute-8C48FC?style=for-the-badge)](https://rai-inst.com/resources/papers/jacta-a-versatile-planner-for-learning-dexterous-and-whole-body-manipulation/)

</div>
Robotic manipulation is challenging and data-driven approaches typically require large amounts of data or expert demonstrations. Therefore, we introduce a motion planner for dexterous and whole-body manipulation tasks. The planner's demonstrations can directly be used by reinforcement learning. With this approach, we can efficiently learn policies for complex manipulation tasks, where traditional reinforcement learning alone only makes little progress.


![Jacta Manipulation](docs/source/_static/images/jacta_overview.jpg)

### Installation
Install cmake
```
sudo apt install cmake
```

Install mujoco-extensions
```
git clone https://github.com/bdaiinstitute/jacta-manipulation.git
cd jacta-manipulation
pip install src/mujoco-extensions -vv
```

Install jacta-manipulation
```
pip install -e .
```

### Getting started
```
cd jacta-manipulation
python jacta-manipulation/examples/planner/example_notebook.py
```


### Citation
```
@inproceedings{brudigam2024jacta,
  author       = {Br{\"u}digam, Jan and Abbas, Ali-Adeeb and Sorokin, Maks and Fang, Kuan and Hung, Brandon and Guru, Maya and Sosnowski, Stefan and Wang, Jiuguang and Hirche, Sandra and Le Cleac'h, Simon},
  editor       = {Agrawal, Pulkit and Kroemer, Oliver and Burgard, Wolfram},
  title        = {Jacta: {A} Versatile Planner for Learning Dexterous and Whole-body Manipulation},
  booktitle    = {Conference on Robot Learning, 6-9 November 2024, Munich, Germany},
  series       = {Proceedings of Machine Learning Research},
  volume       = {270},
  pages        = {994--1020},
  publisher    = {{PMLR}},
  year         = {2024},
  url          = {https://proceedings.mlr.press/v270/bruedigam25a.html},
}
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
