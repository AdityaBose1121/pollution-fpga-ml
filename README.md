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

## Dataset
https://data.mendeley.com/datasets/9rzgv6xd57/1
-This data set includes hourly air pollutant data from   air quality monitoring station of MIT-WPU, Kothrud, Pune, India. The data is collected from (SAFAR-India), System of Air Quality and Weather Forecasting and Research, Ministry of Earth Science, Govt. of India, Indian Institute of Tropical Meteorology, Pune.  The data collected is for the period from 5/06/2019 to 29/06/2022 with 26891 records. The attributes are air pollutants: PM10, PM2.5, PM1, O3, CO, NO, NO2, NOx, CO2.  This data could be very well used for air pollutant analysis, evaluations, calibration of sensors and study of predictive models for air pollutants.

## Model
- Input: 54 features (6-hour window)
- Architecture: 54 → 16 → 8 → 1
- Output: PM2.5 (t+1)

## Fixed-Point Format
- Q8.8 (16-bit)

## Hardware Target
- Basys-3 (Artix-7 FPGA)
