# Bringing your own face data

This repository ships **no face images**. `data/gallery/` and `data/probe/` are
empty by design — see the reasoning at the bottom of this page.

## Layout

Both folders use the same structure: one folder per identity, named exactly as
you want the label to appear.

```
data/
├── gallery/                 # enrolment set — who the system knows
│   ├── Ada Lovelace/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   └── 03.jpg
│   └── Alan Turing/
│       ├── 01.jpg
│       └── 02.jpg
└── probe/                   # held-out set — used only for evaluation
    ├── Ada Lovelace/
    │   └── 04.jpg
    └── Alan Turing/
        └── 03.jpg
```

The folder name is the ground-truth label. Anything that is not a directory is
skipped, so `.gitkeep` files are harmless.

Point the tools somewhere else with flags or environment variables:

```bash
facefuse benchmark --gallery /path/to/gallery --probe /path/to/probe
```

```bash
export FACEFUSE_GALLERY=/path/to/gallery
export FACEFUSE_PROBE=/path/to/probe
```

## What makes a good gallery

* **3–8 images per identity.** More helps, because matching is exhaustive
  against every stored sample — variety matters more than count.
* **Vary the conditions deliberately.** Different lighting, expression and head
  pose across the samples for one person is what the multi-sample gallery is for.
* **Roughly frontal.** The detector finds profile faces, but InsightFace's
  alignment degrades past about 45° of yaw, and the gallery inherits that.
* **Face large enough to survive the crop.** Below roughly 80×80 pixels the
  embedding gets unreliable; the pipeline pads up to 320×320 but cannot invent
  detail.
* **One person per image, or accept the largest-confidence face.** Enrolment
  keeps only the highest-confidence detection per file.

Probe images must be genuinely held out — images that also appear in the gallery
will score a perfect match against themselves and tell you nothing.

## Checking that enrolment worked

`build()` prints a line per identity and warns when an image yields no
embedding:

```
Building hybrid face database...
Scanning: Ada Lovelace
InsightFace failed in 03.jpg
Hybrid Database ready: ['Ada Lovelace', 'Alan Turing']
```

An image can be skipped for two reasons: the YOLO detector found no face above
the confidence threshold, or InsightFace's own alignment stage rejected the crop.
Lower `--threshold` on the detector side or replace the image.

## Why the repository ships empty

The system was developed against a small gallery of public-figure photographs
collected from the web plus photographs of identifiable private individuals.
Neither category can be redistributed:

* the press photographs are copyrighted, and a research project has no licence
  to republish them;
* the photographs of identifiable people are biometric data, and publishing a
  face image next to a name label is exactly the kind of processing that needs
  the subject's consent.

So the evaluation is reproducible in *method* — the code, the parameters and the
metrics script are all here — but not in *data*. Use images you own, or a dataset
whose licence explicitly permits redistribution, and never commit them: the
`.gitignore` already excludes `data/gallery/*` and `data/probe/*` to make the
accident hard to have.
