# Trial Flow

The experiment progresses in **five phases**:

1. **Neutral block**: 40 neutral photos, randomized order, with “Just View” cue.  
2. **Practice block**: 8 unpleasant photos, fixed set, strategy is *just view*.  
3. **Main blocks**: 4 × 40 unpleasant photos, each block randomly assigned one strategy (*view*, *suppression*, *reappraisal*, *suppression+reappraisal*). Both block order and photo order randomized.  
4. **State measures**: After each block, participants complete ratings (7‑state measure, SAM for neutral, plus pain ratings).  
5. **Goodbye routine**: End of experiment.

---

## Per‑Trial Flow

- **Unpleasant trials**:  
  Cross (300 ms) → Cue (1000 ms) → Delay (500–1500 ms) → Photo (4000 ms) → Blank (1000 ms)

- **Neutral trials**:  
  Cross (300 ms) → Cue “Just View” (1000 ms) → Photo (4000 ms) → Blank (1000 ms)

---

# Unpleasant Trials

The section progresses in **four blocks**. Each block consists of:

```
[ reset ] → [ trial (progresses 40 rows at a time) ] → [ state_measure ]
```

This sequence repeats **×4**, covering all 160 randomized trials.

```txt
Trial Flow (4 blocks total):

 ┌──────────┐      ┌──────────────┐      ┌───────────────┐
 │  reset   │  ->  │    trial     │  ->  │ state_measure  ┐
 └──────────┘      └──────────────┘      └───────────────┘│ 
       ▲                  │                               │ 
       │                  │(progresses 40 rows at a time) │ 
       │                                                  │ 
       └──────────────────────────────────────────────────┘
                     repeated ×4
```

This experiment uses a master condition file of 160 trials that is sliced into four sequential blocks of 40. At the start of each block, the reset routine updates the slice indices (row_section), ensuring that only the correct subset of rows is fed into the inner loop. The trial routine then presents each photo stimulus with its assigned emotion regulation technique, while the state_measure routine follows to collect participant ratings. This design keeps the Builder flow modular while guaranteeing randomized block order, randomized photo order within each block, and reproducible logging of the exact slice used per participant.

---

# Code Slicer Properties

The `code_slicer` handles randomization and trial slicing. It lives in the **reset routine**, where it updates the `row_section` variable to select the correct 40‑trial slice from the master condition file. This ensures each block uses a unique randomized subset of trials, while also logging block order and technique assignment for reproducibility.

### Before Experiment
```python
import random
import csv

file_name = 'emotion_regulation_trials.csv'

# Techniques and blocks
techniques = ['view', 'suppression', 'reappraisal', 'suppression-reappraisal']
block_ids = ['block1', 'block2', 'block3', 'block4']

# Photo sets per block
photo_sets = {
    'block1': list(range(1001, 1041)),
    'block2': list(range(2001, 2041)),
    'block3': list(range(3001, 3041)),
    'block4': list(range(4001, 4041)),
}

# Randomize techniques and block order
random.shuffle(techniques)
block_to_technique = dict(zip(block_ids, techniques))
block_order = block_ids.copy()
random.shuffle(block_order)

# Build trial list
trials = []
trial_number = 1
for block in block_order:
    technique = block_to_technique[block]
    photos = photo_sets[block].copy()
    random.shuffle(photos)
    for photo_id in photos:
        trials.append({
            'trial_number': trial_number,
            'block_id': block,
            'technique': technique,
            'photo_filename': f"{photo_id}.jpg"
        })
        trial_number += 1

# Save randomized trial file
with open(file_name, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=trials[0].keys())
    writer.writeheader()
    writer.writerows(trials)

print(f"Saved randomized trial file: {file_name}")
```

---

### Begin Experiment
```python
# Initialize slice indices
start = 0
end = 40

# Store for logging
thisExp.addData('block_order', block_order)
thisExp.addData('block_to_technique', block_to_technique)
```

---

### Begin Routine
```python
# Define slice for current block
row_section = f"{start}:{end}"

# Log which slice was used
thisExp.addData('row_section', row_section)
```

---

### End Routine
```python
# Advance slice for next block
start += 40
end += 40
```

---
