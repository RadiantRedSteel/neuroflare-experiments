# Open–Closed Eyes 2x Experiment
A PsychoPy experiment for alternating eye-state trials with EEG/EMG monitoring.

## Overview
This experiment presents four blocks — two with eyes open and two with eyes closed — to establish a baseline for EEG/EMG recording. It is designed for use with a wired EEG system and standard EMG monitoring device. Participants receive instructions and complete timed fixation trials while alternating eye states.

The experiment uses a randomized loop to determine block order and includes state-measure assessments before and after each block. A parallel-port trigger is sent during each fixation period to synchronize with external hardware.

# State Measures
All state-measure routines use the same configuration.
- Fatigue
- Sleepiness
- Pain
- Pain Unpleasantness
- SAM Valence
- SAM Arousal
- SAM Dominance

These are collected five times: once before trials and once after each block.

# Routines Overview
- `experimentSetup` — initializes screen geometry, wrapWidth logic, and state-measure parameters
- `welcome` — introduces the experiment
- `instruction` — displays the block-specific prompt from the condition file
- `fixation` — displays a fixation cross for 5 minutes with a parallel-port trigger
- `stateMeasure` — collects ratings using the standard configuration
- `goodbye` — end screen with a 4-second exit delay

# Parallel‑Port Trigger
A p_port trigger is sent for the full duration of the `cross_fixation` display in each block. This allows synchronization with EEG/EMG hardware.

# Future Improvements
- Incorporate eye tracking to verify gaze compliance during the fixation period.
