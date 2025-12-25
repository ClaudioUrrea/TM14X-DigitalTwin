# Multimodal Sensor Fusion for Real-Time Fatigue Monitoring in Human-Robot Collaboration: Digital Twin Validation Across 1,000 Simulated Episodes

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fsystems-blue)](https://doi.org/10.3390/systems)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2026a-orange)](https://www.mathworks.com/)
[![RoboDK](https://img.shields.io/badge/RoboDK-5.9.2-red)](https://robodk.com/)

## 📄 Paper Information

**Title:** Multimodal Sensor Fusion for Real-Time Fatigue Monitoring in Human-Robot Collaboration: Digital Twin Validation Across 1,000 Simulated Episodes

**Author:** Claudio Urrea  
**Affiliation:** Electrical Engineering Department, Faculty of Engineering, University of Santiago of Chile  
**Contact:** claudio.urrea@usach.cl

**Journal:** Systems (MDPI)  
**Year:** 2025 
**Status:** Submitted

## Abstract

Sensor fusion for fatigue monitoring is hard. Depth cameras run at 30 Hz, ECG varies with each heartbeat (roughly 0.8–1.2 Hz), and skin conductance samples steadily at 1 Hz—how do you align them? Digital Twins solve one problem: perfect labels. Human subjects can only self-report fatigue (Borg scales, questionnaires), introducing substantial measurement error. Simulation generates 1,000 test episodes from validated cardiovascular and electrodermal models with error-free ground truth from the model’s internal state. The sensor array combines Azure Kinect RGB-D (1920×1080 + 640×576 depth, 0.5–5 m range), Polar H10 ECG (60–200 bpm via Bluetooth LE), Grove GSR (12-bit ADC on finger electrodes), and MediaPipe pose tracking (33 landmarks, 30 fps). Synchronization matters. Aligning timestamps across these different rates, filtering GSR motion artifacts without destroying slow HRV signals, and meeting a 200 ms processing budget for safety decisions—all proved more difficult than expected. Simulated fatigue rises from 15% baseline to 52% after 45 minutes, matching physiological literature. Statistical validation shows large temporal effects: Friedman χ2(3) = 3000.0, p < 0.001, Cohen’s d ranging 1.18 to 2.34. Random Forest classification achieves 94.2% accuracy on fused features versus 76.4% for heart rate alone. That’s nearly 18 percentage points better—about a quarter improvement. A tricolor semaphore (Green/Orange/Red at 30%/35% thresholds) triggers task reallocation when sensor fusion detects high fatigue, achieving 99.30% collision-free operation across all episodes. The ISO/TS 15066:2016 target sits at 99.85%, so there’s still a gap. But vision-only baselines typically hit 95%, making sensor fusion a clear improvement. Results suggest 30–35% reduction in musculoskeletal injury risk becomes feasible through sensor-informed task allocation.

### Key Features

Key Features
Sensor Fusion Performance

94.2% classification accuracy using Random Forest fusion of 28 features (16 physiological + 12 vision-derived)
+17.8 percentage point improvement over best single-sensor baseline (HR only: 76.4%)
Feature importance breakdown: Physiological sensors 58%, vision sensors 27%, contextual features 15%

Real-Time Safety Control

99.30% collision-free operation across 1,000 episodes (approaching ISO/TS 15066:2016 target of 99.85%)
200ms total latency: sensor read (50ms) + sync (30ms) + feature extraction (80ms) + RF inference (20ms) + commands (20ms)
Tricolor semaphore alert system: Green ≤30%, Orange 31-35%, Red >35% fatigue thresholds
Dynamic task reallocation triggers when fused fatigue estimate exceeds 35%

Multimodal Sensor Integration

Azure Kinect RGB-D: 1920×1080 + 640×576 depth, 0.5-5m range, 30 Hz
Polar H10 ECG: HR/HRV metrics via Bluetooth LE, ~1 Hz
Grove GSR: 12-bit ADC skin conductance, 1 Hz
MediaPipe Pose: 33 landmarks, 30 fps skeletal tracking

Statistical Validation at Scale

1,000 simulated episodes with perfect ground truth from Digital Twin (vs. typical HRC studies: n=10-50)
Temporal fatigue progression: 15% baseline → 52% at 45 minutes (Friedman χ²(3)=3000.0, p<0.001)
Large effect sizes: Cohen's d ranging 1.18 to 2.34 across time bins
Fatigue-skill paradox: Direct ρ=0.94 (p<0.001), partial ρ=0.12 (p=0.74, n.s.) controlling for time

Intervention Effectiveness

Post-reallocation physiological recovery: HR -8.3 bpm, HRV RMSSD +12.1 ms, GSR -0.9 µS (all p<0.001, Cohen's d=0.54-1.02)
67.2% of episodes triggered reallocation (mean: 3.2 reallocations per episode)
30-35% reduction in musculoskeletal injury risk estimated through sensor-informed task allocation

Digital Twin Integration

RoboDK simulation with Omron TM14X robot models
MATLAB interface for trajectory generation and control
Virtual testing before physical deployment with synthetic sensor noise validation


## Quick Start

### Prerequisites

```bash
Python 3.8+
NumPy >= 1.20.0
Pandas >= 1.3.0
Matplotlib >= 3.4.0
Seaborn >= 0.11.0
SciPy >= 1.7.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ClaudioUrrea/TM14X
cd TM14X

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy

# Run simulation
python Dynamic_Threshold_Algorithm.py
```

### Running the Simulation

```python
from Dynamic_Threshold_Algorithm import run_simulation

# Run with default parameters (1000 episodes)
run_simulation(num_episodes=1000, verbose_frequency=100)
```

## Results

### Safety Performance

| Metric | Value | Target | Status |
|--------|-------|--------|---------|
| Collision-free rate | 99.30% | 99.850% | APPROACHING TARGET |
| Safe episodes | 993/1000 | - | - |
| Total interventions | 2011 | - | - |

### Fatigue Evolution

Time Point | Mean Fatigue | Std Dev | Semaphore State | n
-----------|--------------|---------|-----------------|------
t=0 min    | 28.06%       | 0.74%   | 🟠 ORANGE       | 1000
t=15 min   | 38.07%       | 0.96%   | 🟠 ORANGE       | 1000
t=30 min   | 52.00%       | 1.29%   | 🔴 RED          | 1000
t=45 min   | 68.04%       | 1.51%   | 🔴 RED          | 1000

### Statistical Validation

Statistical Validation
Temporal Effects Analysis

Friedman Test (non-parametric repeated measures): χ²(3) = 3000.0, p < 0.001
Fatigue progression: 15.2% baseline → 51.9% at 45 min (exponential accumulation, ρ = 0.99, p < 0.001)
Effect sizes (Cohen's d): 1.18 to 2.34 across time bins (very large effects)
Significance threshold: α = 0.001 (Bonferroni-corrected for multiple comparisons)

Fatigue-Skill Paradox

Direct correlation: ρ = 0.94, p < 0.001 (strong positive)
Partial correlation (controlling for time): ρ_partial = 0.12, p = 0.74 (not significant)
Skill progression: Linear improvement (ρ = 0.97, p < 0.001)
Key finding: Skill improvement does NOT mitigate physiological fatigue—both increase with time

Classification Performance Metrics

Overall accuracy: 94.2% (n = 13,500 samples from 300 test episodes)
Per-class F1-scores: Green 0.953, Orange 0.908, Red 0.946
Weighted average: Precision 0.941, Recall 0.942, F1 0.941

Inter-Sensor Correlations (Spearman ρ)

HR vs. HRV: ρ = -0.76* (p < 0.001)
HR vs. GSR: ρ = 0.68* (p < 0.001)
HRV vs. GSR: ρ = -0.61* (p < 0.001)
Correlation range: |ρ| = 0.48–0.76 (moderate-to-strong, justifying multimodal fusion)

Post-Reallocation Recovery (n = 3,187 events)

Heart Rate: -8.3 bpm (Cohen's d = 0.72, p < 0.001)
HRV RMSSD: +12.1 ms (d = 0.68, p < 0.001)
GSR: -0.9 µS (d = 0.54, p < 0.001)
Fatigue state: -11.8% (d = 1.02, p < 0.001)

Sensor-Specific Temporal Changes

HR increase: 72 → 94 bpm (+30.5%, p < 0.001)
HRV decrease: 58 → 31 ms RMSSD (-46.6%, p < 0.001)
GSR increase: 2.3 → 4.1 µS (+78.3%, p < 0.001)
Posture degradation: Shoulder angle 22° → 35° (+59.1%, p < 0.001)

## Repository Structure

```
fatigue-aware-hrc/
│
├── Dynamic_Threshold_Algorithm.py     # Main simulation script
├── simulation_data.csv                # Raw simulation data (1000 episodes)
├── simulation_summary.json            # Statistical summary
│
├── visualizations/
│   ├── collision_analysis.png         # Safety performance analysis
│   ├── fatigue_trajectory.png         # Fatigue evolution over time
│   ├── semaphore_distribution.png     # Alert state distribution
│   └── skill_vs_fatigue.png           # Fatigue-skill paradox
│
├── README.md                          # This file
├── LICENSE                            # CC BY 4.0 License
└── CITATION.cff                       # Citation metadata
```

## Methodology

### Fatigue Monitoring System

The system implements a **tricolor semaphore alert system**:

- 🟢 **GREEN (0-30%)**: Safe zone - Normal sampling (5 min)
- 🟠 **ORANGE (31-40%)**: Caution zone - Increased monitoring (2 min)
- 🔴 **RED (>40%)**: Critical zone - Real-time monitoring (0.1 min)

### Task Configuration

| Task | Duration | Fatigue Rate | Physical Demand | Base Fatigue |
|------|----------|--------------|-----------------|--------------|
| Rest Period | 15 min | 0.0%/min | Very Low | 15% |
| HMI Supervision | 15 min | 0.87%/min | Low | 15% |
| Material Delivery | 15 min | 0.67%/min | Moderate-High | 15% |
| Packaging Preparation | 15 min | 0.93%/min | Very High | 15% |

### Hardware Configuration

- **Robots:** 2× Omron TM14X collaborative robots (14 kg payload, 1100 mm reach)
- **Sensors:** Azure Kinect DK, Polar H10 (HR/HRV), Grove GSR sensor
- **Workstation:** AMD Ryzen 9 5950X, 64 GB RAM, NVIDIA RTX 4080 16GB
- **Software:** MATLAB 2026a, RoboDK 5.9.2, Python 3.8+

## Visualization Examples

### Fatigue Trajectory
![Fatigue Trajectory](fatigue_trajectory.png)

### Collision Analysis
![Collision Analysis](collision_analysis.png)

### Semaphore Distribution
![Semaphore Distribution](semaphore_distribution.png)

### Skill vs Fatigue
![Skill vs Fatigue](skill_vs_fatigue.png)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{urrea2025fatigue,
  title={Multimodal Sensor Fusion for Real-Time Fatigue Monitoring in Human-Robot Collaboration: Digital Twin Validation Across 1,000 Simulated Episodes},
  author={Urrea, Claudio},
  journal={Systems},
  volume={1},
  number={1},
  pages={0},
  year={2026},
  publisher={MDPI},
  doi={10.3390/systems}
}
```

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

**Claudio Urrea**  
Electrical Engineering Department  
Faculty of Engineering  
University of Santiago of Chile  
Email: claudio.urrea@usach.cl  
ORCID: [0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)

## Acknowledgments


RoboDK provided an educational license that made the high-fidelity simulation possible. The Faculty of Engineering at Universidad de Santiago de Chile supplied computational resources and lab facilities. Thanks to the anonymous reviewers whose feedback improved the manuscript substantially.

## Data Availability

- **Simulation Data:** Available in this repository (`simulation_data.csv`)
- **Full Dataset:** Available on FigShare at https://doi.org/10.6084/m9.figshare.30520922
- **Code:** Available on GitHub at https://github.com/ClaudioUrrea/TM14X-DigitalTwin

## Related Publications

1. Urrea, C. (2026). "Multimodal Sensor Fusion for Real-Time Fatigue Monitoring in Human-Robot Collaboration: Digital Twin Validation Across 1,000 Simulated Episodes" *Systems*, MDPI.

---

**Keywords:** sensor fusion; multimodal sensing; RGB-D sensors; physiological sensors; wearable sensors; sensor synchronization; real-time sensor processing; sensor data integration; fatigue monitoring sensors; human-robot collaboration; Azure Kinect; heart rate sensors; galvanic skin response; pose estimation sensors; Digital Twin; sensor-based control


**MSC:** 93C85; 93D21; 93C40; 90C29; 90B50; 68T40; 68T05; 49J15; 93B52

---

*Last updated: December 2025*
