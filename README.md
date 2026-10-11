# Turbo-Transformer

A from-scratch implementation of a **~55M-parameter Transformer language model** built using PyTorch, exploring the core building blocks of modern language models—from subword tokenization and embeddings to Transformer-based sequence modeling.

## Overview

Turbo-Transformer is a hands-on exploration of the architecture and computational principles behind Transformer-based language models. The project focuses on implementing fundamental components and understanding how text is transformed into token representations suitable for neural sequence modeling.

## Key Components

- **Transformer Language Model:** A ~55M-parameter model implemented using PyTorch.
- **Byte Pair Encoding (BPE):** Subword tokenization through iterative merging of frequent token pairs.
- **Tokenization:** Experiments with tokenizer implementations and separate English and Italian vocabularies.
- **Embeddings:** Implementations for mapping discrete tokens into continuous vector representations.

## Tech Stack

- **PyTorch** — Neural network implementation and tensor computation
- **NumPy** — Numerical operations
- **Pandas** — Data processing and analysis
- **Einops** — Readable tensor transformations

## Repository Structure

The repository includes tokenizer and embedding implementations, BPE tokenizer notebooks, and language-specific vocabulary files.

## Motivation

The goal of this project is to understand and implement the fundamental mechanisms underlying modern language models, with an emphasis on learning through implementation rather than treating the Transformer as a black box.