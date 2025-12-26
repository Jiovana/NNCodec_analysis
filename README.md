# NNCodec_analysis

Scripts and experimental results for the analysis of neural network compression using NNCodec and DeepCABAC, with a focus on compression efficiency, reconstruction error, execution time, and memory usage.

This repository accompanies the paper submitted to Signal Processing: Image Communication (special issue on learned visual data coding), and is intended to support reproducibility, result inspection, and further experimentation.

## Overview

The experiments in this repository evaluate entropy-coded, quantized neural networks using the MPEG Neural Network Compression (NNC) standard and its reference implementation, NNCodec.

The analysis covers:

Compression efficiency across multiple neural architectures

Reconstruction fidelity (e.g., NMSE) under quantization and entropy coding

Parameter-wise and layer-wise compression behavior

Execution time and peak memory usage of the DeepCABAC entropy coder

Repository Structure

The repository is mainly organized as follows:

- NNCodec_analysis/
  - deepcabac compression scripts/
    - Scripts for running DeepCABAC tensor-wise encode/decode flow
    - Generates compression statistics and saves reconstructed model
    - Contains the unprocessed tensor-wise results for each model
  - nncodec flow compression/
    - Scripts for running NNCodec encode/decode flow
    - Includes full-model compression 
  - profiler code/
    - Profiling results and scripts used to parse profiling data 
    - Actual Profiler.h file utilized with DeepCABAC
    - Profiling was performed using the compression scripts
  - plot and analysis scripts/
    - Data processing and statistical analysis scripts
    - Generates tables, summaries, intermediate metrics and figures
    - Contains the respective results for each model
  - inference scripts/
    - Scripts to perform inference on evaluation datasets for both baseline and reconstructed models
  - test and other scripts/
    - Auxiliary scripts
