# DHSP-3W: Three-Way Decision for Selective Visual Question Answering

[![Paper](https://img.shields.io/badge/paper-IEEE%20T--AFFC-blue)](paper/main.pdf)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)

Reference implementation of the paper:

> **Three-Way Decision for Selective Visual Question Answering: Hedge Absorption Expands Coverage at Fixed Confident-Error Budget**
> Anup Batakurki, Ramesh Chundi
> School of Computer Science and Engineering, Dayananda Sagar University

---

## Table of contents

1. [Motivation](#motivation)
2. [Method in one paragraph](#method-in-one-paragraph)
3. [Key results](#key-results)
4. [What this repository is not](#what-this-repository-is-not)
5. [Installation](#installation)
6. [Data](#data)
7. [Quick start](#quick-start)
8. [Repository layout](#repository-layout)
9. [Reproducing paper tables](#reproducing-paper-tables)
10. [Reproducing paper figures](#reproducing-paper-figures)
11. [Configuration](#configuration)
12. [Evaluation protocol](#evaluation-protocol)
13. [Limitations](#limitations)
14. [Citation](#citation)
15. [Acknowledgments](#acknowledgments)
16. [Contact](#contact)
17. [License](#license)

---

## Motivation

Selective prediction for assistive VQA usually forces a binary answer-or-abstain decision under one error budget. That framing treats two error types as equivalent when they are not.

In VizWiz, a wrong answer from a VQA model can take one of two forms:

- The model outputs `unanswerable`. A blind user retries, rephrases, or asks someone else. This is a **hedged error (H-error)** and it is benign.
- The model outputs a confident wrong answer. The user may act on it — take the wrong pill, cross the wrong street. This is a **confident error (C-error)** and it is the safety-relevant failure mode.

Prior work counts both as "error" and reports a single budget `P(error) ≤ ε`. It also counts hedged outputs as answers, which inflates coverage. We correct both.

---

## Method in one paragraph

Given a calibrated C-error probability `p_C(e)` and a hedged-error probability `p_H(e)`, DHSP-3W applies a cost-sensitive argmax over three actions:
