# MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time
![banner](figs/banner.png)

## Overview
**MiGrAte** is a test-time adaptation framework that iteratively searches for optimal solutions in challenging domains. Given a search problem, MiGrATe iteratively searches for optimal solutions by sampling candidates and updating its policy model $\pi_\theta^t$ using mixed-policy GRPO.
    In each iteration, we combine online samples ($\bullet$) from the current policy distribution,
    % are generated and a 
    top-performing past solutions ($\star$) as greedy references,
    and samples drawn from the neighborhoods of greedy solutions ($\circ$) to form a GRPO group. 
    The resulting group is used to update $\pi_\theta^t$ and *migrate* towards a sampling distribution that is likely to generate higher-quality solutions according to $f(\cdot)$.

## Environment Setup
```bash
conda create -n migrate python=3.10
conda activate migrate  
pip install -r requirements.txt
```

## Running Experiments

### Semantle (Word Search)
**Run Scripts**
```bash
# Baselines
python semantle_inference.py --strat "random_sample"       # Random
python semantle_inference.py --strat "random_top3_ns"      # NS
python semantle_inference.py --strat "opro_10k"            # OPRO
./scripts/semantle/1_grpo.sh                               # GRPO
./scripts/semantle/2_grpo_greedy.sh                        # GRPO-Greedy
# MiGrAte Variants
./scripts/semantle/3_migrate.sh                            # MiGrAte
./scripts/semantle/4_migrate_opro.sh                       # MiGrAte (OPRO)
```


## Dockstring (Molecule Optimization)
- Install [Open Babel](https://openbabel.org/docs/Installation/install.html)

**Run Scripts**
```bash
# Random
python dockstring_inference.py --strat "random_sample"
# NS
python dockstring_inference.py --strat "random_top3_ns"
# OPRO
python dockstring_inference.py --strat "opro_5k"
# GRPO
./scripts/dockstring/1_grpo.sh
# GRPO-Greedy
./scripts/dockstring/2_grpo_greedy.sh
# MiGrAte
./scripts/dockstring/3_migrate.sh
# MiGrAte (OPRO)
./scripts/dockstring/4_migrate_opro.sh
```

## ARC (Abstraction and Reasoning Corpus)
- Dataset Setup:  
    - Ensure the following are installed:
      - Kaggle API (`kaggle` package)
    - Set up Kaggle API credentials (if not already done):
      - Go to [Kaggle](https://www.kaggle.com/).
      - Navigate to **Account Settings**.
      - Scroll down to the **API** section and click "Create New API Token".
      - This downloads a `kaggle.json` file.
      - Move this file to `~/.kaggle/`.
      - Set appropriate permissions:
        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```
    - Download the dataset:
       ```bash
       kaggle competitions download -c arc-prize-2024 -p ./kaggle/input/arc-prize-2024
       unzip ./kaggle/input/arc-prize-2024/arc-prize-2024.zip -d ./kaggle/input/arc-prize-2024/
       ```
       
### ARC-Small/Full scripts
```bash
# ARC-Small
./scripts/arc_small/0_random.sh           # Random
./scripts/arc_small/1_ns.sh               # NS
./scripts/arc_small/2_opro.sh             # OPRO
./scripts/arc_small/3_grpo.sh             # GRPO
./scripts/arc_small/4_grpo_greedy.sh      # GRPO-Greedy
./scripts/arc_small/5_migrate.sh          # MiGrAte
./scripts/arc_small/6_migrate_opro.sh     # MiGrAte (OPRO)

# ARC-Full
./scripts/arc_full/0_random.sh            # Random
./scripts/arc_full/1_ns.sh                # NS
./scripts/arc_full/2_opro.sh              # OPRO
./scripts/arc_full/3_grpo.sh              # GRPO
./scripts/arc_full/4_grpo_greedy.sh       # GRPO-Greedy
./scripts/arc_full/5_migrate.sh           # MiGrAte
./scripts/arc_full/6_migrate_opro.sh      # MiGrAte (OPRO)
```
