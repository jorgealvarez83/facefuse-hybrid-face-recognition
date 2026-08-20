# Architecture

FaceFuse is a linear pipeline with one branch point: after the face crop is
enhanced, two independent feature extractors run on it, and their similarity
scores are combined by a weight derived from the crop's sharpness.

```
                    ┌──────────────────────────┐
  frame / image ───▶│ FaceDetector (YOLO11-L)  │  conf ≥ 0.55, single class "face"
                    └────────────┬─────────────┘
                                 │ bbox + 20% margin
                    ┌────────────▼─────────────┐
                    │ DIEMPlus (bilateral)     │  edge-preserving denoise
                    └────────────┬─────────────┘
                                 │ enhanced crop
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼────────┐  ┌──────▼───────┐  ┌───────▼────────────┐
   │ pad_image 320×320 │  │ LBP  (256-d) │  │ Laplacian variance │
   │ InsightFace       │  │ HOG (3780-d) │  │  = sharpness       │
   │  → 512-d embedding│  │  → 4036-d    │  └───────┬────────────┘
   └──────────┬────────┘  └──────┬───────┘          │
              │                  │                  │
     cosine vs gallery   cosine vs gallery      w_deep = clip(var/scale, 0, 1)
              │                  │                  │
              └────────┬─────────┴──────────────────┘
                       │
              fused = w_deep·S_deep + (1 − w_deep)·S_hand
                       │
              argmax over gallery; below threshold → "Unknown"
```

## Modules

| Module | Responsibility |
| --- | --- |
| `facefuse/config.py` | Repository-relative paths and every tunable default. |
| `facefuse/face_detection.py` | `FaceDetector` (YOLO wrapper) and `pad_image` letterboxing. |
| `facefuse/enhancement.py` | `DIEMPlus` bilateral filter with configurable `d`, σ_color, σ_space. |
| `facefuse/recognition.py` | `FaceRecognizer` (InsightFace), `cosine_similarity`, `identify`. |
| `facefuse/database.py` | `FaceDatabase` — deep-only gallery, used by the baseline. |
| `facefuse/hybrid_recognition.py` | LBP+HOG extractor, hybrid gallery, quality-adaptive matcher. |
| `facefuse/genetic_optimizer.py` | Real-valued GA: truncation selection, arithmetic crossover, Gaussian mutation. |
| `facefuse/evaluation.py` | Accuracy / precision / recall / F1 / confusion matrix / FPS. |
| `facefuse/benchmarking.py` | Runs both recognisers and prints the comparison. |
| `facefuse/visualization.py` | Figures: accuracy, confusion matrix, ROC, fusion weights. |
| `facefuse/pipeline.py` | `facefuse benchmark` — enrol, optimise, evaluate, plot. |
| `facefuse/realtime.py` | `facefuse webcam` — live loop. |
| `facefuse/annotate.py` | `facefuse annotate` — still image in, annotated image out. |
| `facefuse/cli.py` | Dispatches the three subcommands. |

## Why this design

**Detector and recogniser are separate models.** InsightFace ships its own
detector, but a YOLO11-L fine-tuned on WIDER FACE localises small and
partially occluded faces more reliably in wide frames. The detector produces
the crop; InsightFace only embeds it. The 20% margin exists because InsightFace's
internal alignment step needs facial context beyond the tight box — a tight crop
frequently makes `app.get()` return nothing at all.

**Fusion at score level, not feature level.** The two descriptors have
incomparable geometry: a 512-d embedding trained with an angular margin loss,
and a 4036-d concatenation of a normalised histogram with gradient bins.
Concatenating them would let the longer, unnormalised vector dominate the cosine.
Comparing them separately and mixing the two *scalars* keeps each similarity in
its own well-behaved space.

**Laplacian variance as the gate.** It is a single convolution, costs nothing
next to two neural forward passes, and degrades in the same direction as the
embedding does: motion blur and defocus destroy both the high-frequency detail
LBP/HOG encodes and the fine texture the embedding relies on — but the embedding
fails *silently and confidently*, which is the dangerous mode. The weight is a
deliberate, inspectable fallback.

**Genetic search, not grid search.** Five parameters, a non-differentiable
objective (top-1 accuracy is piecewise constant), and interactions between the
enhancement parameters and the recognition threshold. A GA needs no gradient and
handles the mixed continuous/integer space directly.

## Data flow during enrolment

`HybridFaceDatabase.build()` walks `data/gallery/<identity>/`, and for each image
keeps the highest-confidence detection only. Each identity maps to a list of
`(embedding, handcrafted)` tuples — one per usable image. Matching is exhaustive:
every probe is compared against every stored tuple. This is O(gallery size) per
query, which is fine for tens of identities and is the first thing to replace for
larger deployments (see *Roadmap* in the README).
