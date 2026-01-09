# neuroflare-experiments
PsychoPy experiments and orchestration scripts for EEG/EMG pain flare-up research

## Repository Layout
```
neuroflare-experiments/
├─ neuroflare/
│   ├─ experiments/        # individual experiment folders
│   │   ├─ open-closed/    # example experiment
│   │   │   ├─ open_closed.psyexp
│   │   │   └─ data/       # participant data outputs
│   │   └─ another-experiment/
│   │
│   ├─ shared/             # reusable assets
│   │   ├─ loop-templates/ # trial loop condition tables
│   │   ├─ pictures/       # SAM images, stimuli, etc.
│   │   └─ stimuli/        # placeholder for audio/video stimuli
│   │
│   ├─ scripts/            # utility scripts (e.g., data organizer)
│   ├─ drivers/            # parallel port DLLs or hardware drivers
│   └─ config/             # global experiment settings (YAML/JSON)
│
├─ analysis/               # notebooks, statistical scripts, visualization
├─ docs/                   # IRB protocols, experiment notes, contributor guide
└─ README.md
```

---

## Current Experiments

### Open‑Closed
A minimal visual‑attention experiment used for baseline recordings and workflow validation. Participants alternate between looking at a fixation cross for five minutes and resting with no visual target for five minutes.

Locations:  
- `neuroflare/experiments/open-closed/`  
- `neuroflare/experiments/open-closed-2x/` (two‑cycle version)

### PHODA Picture‑Viewing Task 
A two‑block experiment presenting daily‑activity photographs from the OpenPHODA‑Short set. One block uses a 0–100 VAS slider to rate perceived harmfulness; the other is passive viewing only. Both blocks use randomized image order, include pre‑ and post‑block state measures, and send image‑aligned parallel‑port triggers for EEG/EMG synchronization.

Location: `neuroflare/experiments/phoda/`

### Emotion‑Regulation
A full experimental task involving neutral and unpleasant images, block‑wise emotion‑regulation strategies, randomized trial slicing, and multiple state‑measure routines.
Includes parallel‑port triggers for EEG/EMG synchronization and dynamically generated condition files.

Location: `neuroflare/experiments/emotion-regulation/`

## Getting Started
1. **Clone the repo**  
   ```bash
   git clone https://github.com/RadiantRedSteel/neuroflare-experiments.git
   ```

2. **Install dependencies**  
   - PsychoPy (latest stable release)  
   - Python 3.10.19  
   - Any hardware drivers in `neuroflare/drivers/`
   - To recreate the Conda environment: ```conda env create -f environment.yml```

3. **Run an experiment**
   - Navigate to `neuroflare/experiments/<experiment-name>/`
   - Open the `.psyexp` file in PsychoPy Builder
   - Ensure any required drivers (e.g., parallel port DLLs) are present
   - Run the experiment from Builder

## Shared Assets
- **Loop templates**: Reusable condition tables for trial structures  
- **Pictures**: SAM images and other visual stimuli  
- **Stimuli**: Placeholder for audio/video assets  
- **Scripts**: Utilities for organizing data folders or batch processing outputs  

## Data Management
- Each experiment writes participant data into its own `data/` folder  
- Utility scripts in `neuroflare/scripts/` can be used to reorganize or archive data across experiments  
- Use relative paths to reference shared assets (`../shared/pictures/...`) for reproducibility

## Hardware Notes
- Parallel port drivers must be in the same directory as the `.psyexp` file  
- Alternatively, symlinks or shortcuts can be used to reference `neuroflare/drivers/`  
- Document any hardware setup or program notes in `docs/`

## Contributing
- Add new experiments under `neuroflare/experiments/`  
- Place shared assets in `neuroflare/shared/`  

## Useful Links
[Parallel Port Issues w/ Windows 11](https://discourse.psychopy.org/t/parallel-port-issues-w-windows-11/45464/19)

[PsychoPy Builder Execution Order Logic Rules](docs/builder_execution_order.md)

---

*Experiments tested with PsychoPy Builder v2025.2.3beta*
