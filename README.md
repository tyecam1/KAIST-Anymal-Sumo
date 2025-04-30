# ME491 Anymal Sumo – Learning-Based Control Project

This project was developed during the **ME491: Learning-Based Control** course at **KAIST** in 2023. It implements a reinforcement learning-based quadrupedal robot trained in simulation to compete in a sumo wrestling environment.

The robot uses the **Proximal Policy Optimization (PPO)** algorithm in a **RaiSim** simulation environment, with a curriculum-shaped reward function encouraging mechanical stability and aggressive engagement.

## Repository Structure

ME491-Anymal-Sumo/ ├── me491_project/ # Main code (algo, env, data, helper)
├── rsc/ # URDF and DAE robot model files
├── third_party/ # External dependencies (RaiSim, RaisimGym)
├── report/ # Final report
└── scripts/ # (Optional) Training/testing scripts


## How to Use
```bash
### 1. Clone This Repo (with submodules)

git clone --recurse-submodules https://github.com/<your-username>/ME491-Anymal-Sumo.git
cd ME491-Anymal-Sumo

git submodule update --init --recursive

### 2. Build RaiSim

cd third_party/raisimLib
mkdir build && cd build
cmake .. && make -j4

### 3. Train an Agent

cd me491_project
python algo/ppo/runner.py --cfg data/<your_experiment_folder>/cfg.yaml

## Key Features
Custom Training Environment based on RaiSim

Curriculum Learning encoded via reward shaping

Stable PPO Implementation for continuous action control

Modular Structure for training, testing, evaluation, and policy storage

## Documentation
The full technical explanation and analysis are available in docs/TyeCameronFinalReport.pdf.

## Acknowledgements
This project was conducted as part of ME491: Learning-Based Control at KAIST, under the supervision of Prof. Jemin Hwangbo.

Special thanks to the following resources:

RaiSim Gym Tutorial – © Jemin Hwangbo

RaiSim Physics Engine – © Jemin Hwangbo

2023 KAIST ME491 Student Projects

Please note: All third-party components remain under their respective licenses.

## License
This project is released under the MIT License (see LICENSE file). Third-party libraries (RaiSim, RaiSimGym) retain their own respective licenses and are included here via Git submodules for educational purposes only.

