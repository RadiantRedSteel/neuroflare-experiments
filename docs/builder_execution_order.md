# PsychoPy Builder Execution Order Logic Rules

## 1. Begin‑Routine code that initializes values used by component parameter updates must be placed *above* those components.
**Why it matters:**
During the Begin‑Routine phase, Builder first runs the **parameter update block**, where it executes lines like:
- `image.setImage(photo_filename)`
- `text.setText(label)`
- `stim.setSize([w, h])`

This happens **before** Begin‑Routine code components run — unless your Code Component is placed *above* the component in the Builder list.

So if your Begin‑Routine code computes:
- the image path
- the text to display
- the geometry
- the sound file
- the position

…and that code runs **after** the component update block, the component will receive an undefined or stale value. That’s where the “sometimes it crashes, sometimes it doesn’t” behavior comes from.

## 2. Begin‑Routine code that computes values used only during Each‑Frame logic can be placed anywhere.
**Why it matters:**
Values like:
- durations
- jitter
- flags
- timers
- thresholds

…are not used during the Begin‑Routine parameter update block. They’re only referenced later during the Each‑Frame loop.

So even if the Code Component runs after the components, it doesn’t matter — Builder won’t use those values until the first frame.

This is why my duration‑calculation example was safe even when the Code Component wasn’t at the top.

## 3. Each‑Frame code must be placed *below* the components whose status or properties it depends on.
**Why it matters:**
Builder processes Each‑Frame logic **top‑to‑bottom**.
So if your Code Component appears *above* the component it references, the sequence becomes:
1. Your code runs
2. Builder updates the component’s status
3. Builder draws the component
4. Flip

This means your code is always reacting to the *previous* frame’s status.

I had an issue where background_box was one frame out of sync.
My code was checking:

```python
if image_iasp.status == FINISHED:
    background_box.autoDraw = False
```

…but because the Code Component was above the image, the status hadn’t been updated yet.
Moving the Code Component *below* the image fixed the timing instantly.

This rule is essential for:
- visibility sync
- trigger timing
- autoDraw toggles
- frame‑accurate stimulus transitions
- conditional triggers

## 4. End‑Routine code fires when the *last active component* ends, not when the routine ends.
**Why it matters:**
Builder ends the component clock early if all components are finished, even if the routine has a fixed duration.

So End‑Routine code may fire:
- at 4 seconds
- even if the routine lasts 5 seconds

For instance, my background_box End‑Routine logic fired at 4 seconds, even though the routine continued for another second of blank screen.

This can lead to some confusion if you're not expecting the code to happen early.

## 5. Code Components that depend on other Code Components must be ordered correctly.
**Why it matters:**
Begin‑Routine code runs in the order the Code Components appear in the Builder list.

So if you have:
- Code A (defines `stim_path`)
- Code B (uses `stim_path`)

…and you move Code A below Code B, Code B will crash or use stale values.

This becomes especially important when you have:
- multiple “before experiment” blocks
- multiple Begin‑Routine helpers
- shared config loaders
- category resolvers
- geometry calculators
- trigger initializers

# TLDR Rules

### Begin‑Routine
- If the code computes values used by component parameter updates → **place above those components**.
- If the code computes values used only during Each‑Frame logic → **placement doesn’t matter**.
- If Code Component B depends on Code Component A → **A must be above B**.

### Each‑Frame
- If the code depends on component status or properties → **place below those components**.
- If the code controls visibility or triggers → **place at the bottom**.

### End‑Routine
- Fires when the last active component ends → **not tied to routine duration**.
- Use only for cleanup, not frame‑accurate timing.

---

*2026-01-08: This data was gathered through my own personal testing on PsychoPy Builder v2025.2.3beta* ***~RadiantRedSteel***
