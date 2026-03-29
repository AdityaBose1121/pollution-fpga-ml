# FPGA-Based ML Accelerator for Pollution Prediction

## Objective
Design and implement a hardware-efficient MLP model on FPGA (Artix-7) for predicting PM2.5 levels.

## Pipeline
1. Dataset cleaning
2. Sliding window generation
3. MLP training
4. CPU reference inference
5. Fixed-point quantization
6. FPGA weight export

## Model
- Input: 54 features (6-hour window)
- Architecture: 54 → 16 → 8 → 1
- Output: PM2.5 (t+1)

## Fixed-Point Format
- Q8.8 (16-bit)

## Hardware Target
- Basys-3 (Artix-7 FPGA)