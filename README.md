[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[mkdocs badge]: <https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat>
[mkdocs]: <https://squidfunk.github.io/mkdocs-material>
[version badge]: <https://img.shields.io/pypi/v/brainblocks.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/brainblocks.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/brainblocks>
[gitter]: <https://gitter.im/brainblocks/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
[discussions]: <https://github.com/BrainBlocksAI/brainblocks/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/BrainBlocksAI/brainblocks>
[ci]: <https://github.com/BrainBlocksAI/brainblocks/actions?query=workflow>
[ci badge]: <https://github.com/BrainBlocksAI/brainblocks/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/BrainBlocksAI/brainblocks/actions?query=workflow>
[doc badge]: <https://github.com/BrainBlocksAI/brainblocks/actions/workflows/doc.yml/badge.svg?branch=main>

# brainblocks

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

A Reinforcement Learning library that implements a range of classical, deep, and LLM-based RL algorithms. 

It mimics the simplicity of [scikit-learn's](https://scikit-learn.org/) API and integrates with the
[Tianshou](https://tianshou.org/en/stable/) RL library. It also leverages
[textgrad](https://github.com/huggingface/textgrad), [DSPy](https://github.com/darkprinx/dspy), and
[DocArray](https://github.com/docarray/docarray) to support the implementation LLM-based agents.

## Installation

For user installation, `brainblocks` is currently available on the PyPi's repository, and you can
install it via `pip`:

```bash
pip install brainblocks
```

Development installation requires to clone the repository and then use [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/BrainBlocksAI/brainblocks.git
cd brainblocks
pdm install
```
