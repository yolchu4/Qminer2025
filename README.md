# Process Control Model — Demo Version

## Overview
This project is a **demo-grade, research-aligned machine learning system**
for industrial process control.

The goal is to **infer unknown control parameters**
from known process inputs and outputs,
using nonlinear, noisy, and dynamic time-series data.

The system is designed to be:
- Realistic
- Reproducible
- UI-independent
- Extendable to real industrial use cases

---

## Problem Definition

A process is described by three conceptual groups:

1. **Process Inputs**  
   External or disturbance variables (not directly controllable)

2. **Control Parameters**  
   Decision variables to be inferred by the model

3. **Process Outputs**  
   Observable process measurements

The learning objective is:

> **Infer control parameters given known inputs and outputs**

