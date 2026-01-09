# PHODA Picture-Viewing Task
A PsychoPy experiment for presenting daily-activity photographs (OpenPHODA-Short Electronic Version) with harmfulness ratings and EEG/EMG synchronization.

## Overview
This experiment presents a set of PHODA (Photograph Series of Daily Activities) images across two picture-viewing blocks. Each block contains the same set of photographs, but the order is randomized independently for each run.

One block requires participants to **rate how harmful** they perceive each activity to be using a **0-100 visual analogue scale (VAS)**.  
The other block is a **passive-viewing** run in which participants simply view each image without making a rating.

A parallel-port trigger is sent for the full duration of each image presentation to allow EEG/EMG synchronization.

# Experiment Structure
The experiment consists of **two picture-viewing blocks**, each containing 40 PHODA images:

1. **Randomized ITI (1500-2500 ms)**  
   A cross fixation with a variable delay before each trial.

2. **PHODA Image (6000 ms)**  
   The photograph is displayed for a fixed duration.  
   A parallel-port trigger remains active for the entire image presentation.

3. **Block-Specific Behavior**  
   - **Rate Block:** A custom VAS slider (0-100) appears below the image. Participants provide a harmfulness rating before continuing.  
   - **View Block:** No rating is shown; participants simply view the image for the full duration.

The order of the two blocks is randomized.

# Per-Trial Flow
```
Gray Screen (1500-2500 ms, randomized)
↓
PHODA Image (6000 ms)
↓
When rating: Harmfulness Rating (0-100 VAS)
```

# State Measures
All state-measure routines use the same configuration.
- Fatigue
- Sleepiness
- Pain
- Pain Unpleasantness
- SAM Valence
- SAM Arousal
- SAM Dominance

These are collected three times: once before trials and once after each block.

# Routines Overview
- `experimentSetup` - initializes screen geometry, wrapWidth logic, and state-measure parameters
- `welcome` - introduces the experiment
- `instruction` - displays the block-specific prompt from the condition file
- `phodaDelay` - presents a randomized 1500-2500 ms inter-trial interval before each PHODA image
- `phodaView` - displays the PHODA image, handles rating or passive-view logic, and sends the image-aligned port signal
- `stateMeasure` - collects ratings using the standard configuration
- `goodbye` - end screen with a 4-second exit delay

# Notes
- Both blocks use the same 40 images, but each block randomizes the order independently.  
- The custom slider is implemented in code to support real-time marker updates and consistent layout beneath the image.
- The slider can be set to use integer values instead of VAS.
- Port signaling (`p_port_phoda`) is tied directly to the onset and offset of the image component for precise synchronization.  

