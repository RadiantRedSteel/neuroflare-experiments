# Emotion-Regulation Task
A PsychoPy experiment for unpleasant-image viewing with block-wise emotion-regulation strategies.

## Overview
This experiment presents neutral and unpleasant images while participants use different emotion-regulation strategies. Three condition files drive the randomization of the task: one for the neutral block, one for the practice block, and one for the unpleasant trials. All three are generated at the start of the experiment. The neutral and practice files require minimal randomization (primarily photo order), while the unpleasant-trial file contains 160 photos that require full randomization and structured slicing.

The 160 unpleasant photos are divided into four sets of 40, each dedicated to a specific block (1-4). Details about block-to-photo associations can be found in `docs/emotion_regulation_block_association.xlsx`. Within each block, the photos are randomized, and each block is assigned a randomly selected emotion-regulation strategy.

Before each unpleasant block, the reset routine updates the slice indices and displays the appropriate strategy-specific instructions.

Throughout the experiment, several state-measure routines are used to assess how the participant is feeling at different stages.

A serialPort trigger is sent to the EEG/EMG system for the full duration of each `image_iasp` presentation.

## Experiment Phases
The experiment progresses in three trial phases:

- **Neutral block**:
  40 neutral photos, randomized order, strategy = *view*.

- **Practice block**:
  8 unpleasant photos, randomized order, strategy = *view*.

- **Unpleasant blocks**:
  4 × 40 unpleasant photos.
  Each block is randomly assigned one strategy:
  *view*, *suppression*, *reappraisal*, *suppression+reappraisal*.
  Block order and photo order are randomized.

## Per-Trial Flow
Cross (300 ms) → Cue (1000 ms) → Delay (500-1500 ms) → Photo (4000 ms) → Blank (1000 ms)

## State Measures
The experiment uses two distinct state-measure configurations, depending on the phase.

### 1. Pre-Trial Baseline
Collected once at the start of the experiment.
Includes 7 ratings:
- Fatigue
- Sleepiness
- Pain
- Pain Unpleasantness
- SAM Valence
- SAM Arousal
- SAM Dominance

### 2. Neutral Block State Measure
Used after the neutral block.
Includes:
- Fatigue
- Sleepiness
- Pain
- Pain Unpleasantness
- SAM Valence
- SAM Arousal
- SAM Dominance

### 3. Unpleasant Block State Measure
Used after each unpleasant block.
Includes:
- Emotion-Regulation Success
- Fatigue
- Sleepiness
- Pain
- Pain Unpleasantness
- SAM Valence
- SAM Arousal
- SAM Dominance

## Routines Overview
- `experimentSetup` - global scaling, layout, and helper initialization
- `welcome` - introduction to the experiment
- `instructionNeutral` - neutral-block instructions
- `instructionPractice` - practice-block instructions
- `instructionNegative` - main unpleasant-block instructions
- `reset` - updates slice indices and displays strategy-specific instructions
- `emotionRegulationCue` - handles the full per-trial cue sequence
- `iaspView` - presents each image
- `stateMeasure` - collects ratings using the appropriate configuration
- `goodbye` - end screen with a 4-second exit delay

# Future Improvements
Future improvements will focus on:
- final wording polish
- text formatting
- debug checks
