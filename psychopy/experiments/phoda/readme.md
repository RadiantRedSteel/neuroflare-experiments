# PHODA Picture-Viewing Task
A PsychoPy experiment for rating perceived harmfulness of daily-activity photographs (OpenPHODA-Short Electronic Version).

---

## Overview

This experiment presents a series of PHODA (Photograph Series of Daily Activities) images. Participants rate how harmful they perceive each activity to be on a 0-100 visual analogue scale. The task consists of two runs, each containing 40 images. EEG/EMG is recorded continuously throughout the task.

A condition file may be generated at the start of the experiment to randomize the order of the 40 PHODA images for each run. However, PsychoPy's built in loop randomization could work in its stead. A parallel-port trigger is sent for the full duration of each image presentation.

Before and after each run, participants complete a set of state-measure ratings assessing fatigue, sleepiness, pain, pain unpleasantness, and SAM affective dimensions.

---

# Experiment Structure

The experiment consists of **two identical runs**, each containing:

1. **Pre‑Run State Measures**  
   Participants rate:
   - Fatigue  
   - Sleepiness  
   - Pain  
   - Pain Unpleasantness  
   - SAM Valence  
   - SAM Arousal  
   - SAM Dominance  

2. **PHODA Picture‑Viewing Block (40 trials)**  
   - Gray screen ITI: **1500–2500 ms**, randomized  
   - PHODA image: **6000 ms** or until participant responds  
   - Rating scale: **0 (Not harmful at all) → 100 (Extremely harmful)**  
   - Response via keypad  
   - EEG/EMG trigger active for entire image duration  

3. **Post‑Run State Measures**  
   Same set of ratings as pre‑run.

Run 2 repeats the same structure, but without per-image ratings if the protocol specifies a passive-viewing second run (based on the PDF). If the study uses ratings in both runs, the experiment can be configured accordingly.

---

# Per‑Trial Flow
```
Gray Screen (1500-2500 ms, randomized)
↓
PHODA Image (6000 ms or until response)
↓
Harmfulness Rating (0-100 VAS)
```
