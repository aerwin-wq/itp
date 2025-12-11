
# Harmony Atlas: MIDI Music Analysis and Remixing

## Project Documentation

**Author:** Arlo Erwin
**Date:** November-December 2025  
**Project Duration:** ~3 weeks of active development

---

## What I Did

I developed **Harmony Atlas**, a neurosymbolic AI system that computationally implements David Lewin's transformational music theory for MIDI analysis and generation. The core philosophy is "discovery not prescription" — using compression algorithms to discover musical relationships rather than hardcoding music theory rules.

### Project Goals
1. Achieve 97-100% lossless MIDI compression and reconstruction
2. Discover musical patterns using <500 universal transforms
3. Create an interpretable, editable "gene editor" for music
5. Generalize learned patterns across unseen files
6. Don't prescribe any music theory rules
7. Enable style transfer and cover song detection

### Applications
1. more efficient song editting
2. cover song detection
3. song similarity metrics
4. style transfer
5. harmonic database
6. MIDI compression



---

## How I Did It

### Theoretical Foundation

The system represents music as "DNA in transform space" using irreducible musical primitives:

| #   | Primitive        | Range        | Count |
|-----|------------------|--------------|-------|
| 1   | rhythm_bucket    | 0–15         | 16    |
| 2   | velocity_bucket  | 0–7          | 8     |
| 3   | pitch_interval   | –127 to +127 | 255   |
| 4   | first_pitch      | 0–127        | 128   |

### Training Data

I used a corpus of 1,000 multitrack MIDI Bigband arrangements downloaded from musescore, as well as my own transcriptions transcribed using tools like basic pitch

# Pipeline (v53)

## Overview

This pipeline discovers musical patterns from MIDI corpora using **pitch-agnostic Grammar-based compression** Unlike earlier attempts that encoded pitch class in terminal symbols, v53 uses **TRUE pitch-agnostic normalization** where terminals contain only rhythm and velocity information. Pitch is stored per-occurrence as a transform parameter.

## Re-Pair (short for recursive pairing) 
![Re_pair_example](https://github.com/user-attachments/assets/60c84245-b40d-4f3b-af81-35b4ebf5d0c3)




## Pure Contour Representation

| Version | Terminal Representation | Consequence |
|---------|------------------------|-------------|
| v50/v52 | `(pitch_class, rhythm, velocity)` | Patterns inherit pitch-class specificity |
| **v53** | `(rhythm_bucket, velocity_bucket)` ONLY | All patterns are purely contour-based |

This achieves **optimal compression** because transposed occurrences of the same melodic shape are unified into a single rule.

## Pipeline Phases

### Phase 1: MIDI Loading
- Loads MIDI files from the specified corpus directory using parallel workers
- Extracts tracks with note information (pitch, timing, velocity, duration)
- Records GM program, drum status, and piece metadata per track

### Phase 2: Pure Contour Grammar Construction
- Uses the **Re-Pair algorithm** (GPU-accelerated) to build a hierarchical grammar
- Terminals: `(rhythm_bucket, velocity_bucket)` pairs (N_TERMINALS = rhythm_buckets × velocity_buckets)
- Rules capture: `(pitch_interval, rhythm_bucket, velocity_bucket)` contours
- Stores `first_pitch` per occurrence for reconstruction
- Hierarchical rules connect child patterns with connector intervals

### Phase 3: Pattern Analysis
- Counts total pattern occurrences across corpus
- Identifies multi-piece patterns (patterns appearing in multiple MIDI files)
- Tracks "merged transposition rules" - rules where the same contour appears at different absolute pitches

### Phase 4: Canonical Pattern Extraction
- Converts grammar rules to `FactoredPattern` objects
- Prepares data structures for subsequent transform discovery phases

### Phase 5: Multi-Stage Transform Discovery

#### Phase 5a: MDL Transform Discovery
- Discovers pitch transforms using Minimum Description Length principle
- Finds transforms that explain pattern relationships efficiently

#### Phase 5b: Multi-Factor Transforms (τ, v, d)
- **τ (tau)**: Rhythm/timing transforms
- **v**: Velocity transforms
- **d**: Duration transforms
- Uses MDL to justify each factor transform

#### Phase 5c: TrackDerive Discovery
- Discovers **cross-track arrangement patterns**
- Finds how patterns relate across different instruments in the same piece
- Records instrument relationships (e.g., "melody doubles bass an octave up")

#### Phase 5d: Interval Magnitude Discovery
- Tests whether **diatonic** or **chromatic** interval representation is better
- Uses MDL to determine optimal representation for the corpus

#### Phase 5e: Feature Importance Discovery
- MDL-based discovery of **conditioning variables**
- Identifies which features (instrument, position, etc.) are useful for prediction

#### Phase 5g: Octave Equivalence Discovery
- Tests whether octave equivalence is beneficial for the corpus
- If yes, intervals ±12 are treated as equivalent, saving bits

### Phase 6: Level 3 Meta-Pattern Discovery
- **Horizontal patterns**: Sequences of transforms that repeat (e.g., I→IV→V→I progressions)
- **Orchestration rules**: Vertical slices showing which instruments play together
- Uses GPU Re-Pair on transform sequences

### Phase 7: Checkpoint Saving
- Saves compressed `.npz` file with grammar tensors
- Exports JSON files for patterns, transforms, and metadata

## Output Files

Given `--output checkpoint.npz`, the pipeline produces:

| File | Contents |
|------|----------|
| `checkpoint.npz` | Main checkpoint with grammar tensors |
| `checkpoint_patterns.json` | Pattern rules with occurrences |
| `checkpoint_transforms.json` | Pitch transform vocabulary |
| `checkpoint_multi_factor.json` | τ/v/d transforms |
| `checkpoint_track_derives.json` | Cross-track derivations |
| `checkpoint_feature_importance.json` | MDL-useful conditioning features |
| `checkpoint_meta.json` | Level 3 meta-patterns and orchestration |

## Usage

```bash
python scripts/run_pure_contour_pipeline_v53_backup.py /path/to/midi/corpus \
    --output checkpoint_v53.npz \
    --max-files 100 \
    --max-rules 5000 \
    --min-count 2 \
    --device cuda \
    --workers 4
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `corpus_path` | (required) | Path to directory containing `.mid` files |
| `--output`, `-o` | `checkpoint_v53_pure_contour.npz` | Output checkpoint path |
| `--max-files` | 100 | Maximum MIDI files to process |
| `--max-rules` | 5000 | Maximum grammar rules to discover |
| `--min-count` | 2 | Minimum pair frequency to create rule |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--workers` | 4 | Parallel workers for MIDI loading |

## Dependencies

### Internal Modules
- `scripts/run_factored_pipeline.py`: FactoredTrack, pattern classes
- `grammar/v4/repair_pure_contour.py`: GPU Re-Pair with pure contour
- `scripts/level3_meta_patterns.py`: Meta-pattern discovery
- `discovery/multi_factor_transforms.py`: τ/v/d transforms
- `discovery/track_derive.py`: Cross-track derivation
- `discovery/interval_magnitude.py`: Diatonic/chromatic selection
- `discovery/feature_importance.py`: MDL conditioning discovery
- `discovery/octave_equivalence.py`: Octave equivalence test

### External
- PyTorch (CUDA support recommended)
- NumPy
- MIDI parsing library (via `load_midi_factored`)

## Data Structures

### PureContourGrammar
```python
grammar.n_terminals      # Number of (rhythm, velocity) terminal pairs
grammar.n_rules          # Number of discovered rules
grammar.final_sequence   # Compressed sequence of rule IDs
grammar.rule_contours    # (interval, rhythm_bucket, velocity_bucket) per rule
grammar.rule_children    # (left_child, right_child) for hierarchical rules
grammar.rule_counts      # Frequency of each rule
grammar.rule_occurrences # {rule_id: [occurrences with first_pitch]}
```

### Rule Format (in JSON)
```json
{
  "rule_id": {
    "pitch_intervals": [2, 2, 1],
    "canonical_pitches": [60, 62, 64, 65],
    "rhythm_bucket": 3,
    "velocity_bucket": 2,
    "count": 47,
    "is_hierarchical": false,
    "is_pure_contour": true,
    "occurrences": [
      {
        "piece_id": "song_001",
        "track_id": 0,
        "first_pitch": 60,
        "onset_time": 1920,
        "gm_program": 0,
        "is_drum": false
      }
    ]
  }
}
```

## Performance Notes

- GPU acceleration (CUDA) provides ~10-50x speedup for grammar construction
- Multi-worker MIDI loading parallelizes I/O-bound file parsing
- Memory usage scales with corpus size and `max_rules`
- Typical runtime: 1-5 minutes for 100 MIDI files on GPU

## Comparison with Earlier Versions

| Aspect | v50/v52 | v53 |
|--------|---------|-----|
| Terminal encoding | pitch_class included | rhythm + velocity only |
| Transposition handling | Separate rules per key | Unified via first_pitch |
| Compression ratio | Good | Optimal |
| Pattern reuse | Within pitch class | Across all pitches |




### Algorithm Details

**Pattern Discovery: Re-Pair Grammar Compression**
```python
# Re-Pair finds globally optimal patterns through frequency-based merging
# Input: CDEFG CDEFG GFEDC
# Process: Replace most frequent pair repeatedly
# Result: Hierarchical grammar capturing structure
```

**Sparse Coding: FISTA Optimization**
```python
# Minimize: ||piece - Σ(w[i]*transforms[i])||² + λ||w||₁
# Where λ approximates MDL cost per transform
```

**Generation: PPM* (Prediction by Partial Matching)**
```python
# Variable-order Markov model with Krichevsky-Trofimov smoothing
# Uses 1-5 pattern context with longest-match fallback
# Combined with position, co-occurrence, style, and chord conditioning
```

---

## Problems I Faced (ABDYDB)

### Problem 1: GPU Memory Overflow (42+ GB)

**Symptoms:**
- A100 40GB GPU running out of memory
- Composition testing attempting to allocate 54.54 GB
- System crash during batched operations

**Root Cause:**
Broadcasting tensors created N×N matrices:
```python
# BAD: Creates (6254, 6254, 256, 128) tensor = 512 GB!
targets = piece_tensors.unsqueeze(1)      # (N, 1, 256, 128)
sources_t = sources_t.unsqueeze(0)        # (1, N, 256, 128)
errors = ((targets - sources_t) ** 2)     # OOM!
```

**Solution:**
Chunked processing with incremental error computation:
```python
# GOOD: Process in chunks, only concatenate small error tensors
chunk_size = 8
for i in range(0, n_comps, chunk_size):
    chunk = compositions[i:i+chunk_size]
    results = apply_batch(corpus, chunk)
    errors = ((corpus - results) ** 2).sum(dim=(2, 3))
    all_errors.append(errors)
    del results  # Free memory immediately
    torch.cuda.empty_cache()
```

### Problem 2: Astronomical Reconstruction Errors (~10¹³⁷)

**Symptoms:**
- FISTA optimization producing infinite errors
- Gradients exploding instead of converging

**Root Cause:**
Sign error in gradient computation caused gradient ASCENT instead of descent:
```python
# WRONG
gradient = -2 * D.T @ (x - D @ a) + lambda_param * sign(a)

# CORRECT  
gradient = 2 * D.T @ (D @ a - x) + lambda_param * sign(a)
```

**Solution:**
Fixed gradient sign and added proper canonicalization to eliminate degenerate patterns like T(12)∘T(-12) cycles.

### Problem 3: Low GPU Utilization (15-22%)

**Symptoms:**
- A100 sitting mostly idle
- Composition testing taking 90 seconds instead of expected seconds

**Root Causes Identified:**
1. Creating new TensorTransformLibrary instances 196 times per iteration
2. Inefficient if-elif chains for transform dispatch
3. Sequential composition processing

**Solutions:**
```python
# Level 1 Fix: Reuse transform library
self.transform_lib = TensorTransformLibrary()  # Create once

# Level 2 Fix: Dictionary lookup instead of if-elif
self.transform_dispatch = {
    'T': self.apply_transpose,
    'I': self.apply_inversion,
    'R': self.apply_retrograde,
    # ...
}

# Level 3 Fix: Batched processing
def compose_transforms_batched(corpus, compositions, chunk_size=32):
    # Process multiple compositions in parallel
```

**Result:** 11-18x speedup, 85-95% GPU utilization

### Problem 4: FAISS GPU Version Mismatch

**Symptoms:**
- GPU functions hanging indefinitely
- CUBLAS errors on certain matrix sizes

**Root Cause:**
faiss-gpu 1.7.2 (compiled for CUDA 10.x/11.x) incompatible with system CUDA 12.6

**Solutions Attempted:**
1. ✅ Use IVF index instead of Flat (approximate but faster)
2. ✅ Fall back to CPU FAISS for large indices
3. ✅ Conda environment with matching CUDA version:
```bash
conda create -n lewinian python=3.10 pytorch pytorch-cuda=12.1 faiss-gpu \
    -c pytorch -c nvidia -c conda-forge
```

### Problem 5: "Locally Good, Globally Random" Generation

**Symptoms:**
- Individual patterns sound correct
- Multi-track output sounds like "cacophony"
- Instruments don't coordinate

**Root Cause:**
Re-Pair creates a deterministic Straight-Line Program suitable for compression but not generation. Generation requires probabilistic sampling over alternatives.

**Solution:**
Implemented 6-level probabilistic sampling system:
```
Level 1 (40%): PPM* - Context-aware pattern selection
Level 2 (15%): Position - Structural awareness  
Level 3 (20%): Co-occurrence - Instrument relationships
Level 4 (15%): Style - Genre consistency
Level 5 (10%): Chord Context - Harmonic grounding
Level 6: Short-term Memory Boost - Pattern reuse at phrase boundaries
```

### Problem 6: Wrong Track/Instrument Mapping

**Symptoms:**
- Orchestration rules returning wrong instrument pairs
- Track lookup failures

**Root Cause:**
Using loop index instead of actual MIDI track_id:
```python
# WRONG: Uses loop index (0, 1, 2, 3)
for i, track in enumerate(valid_tracks):
    occurrence['track_id'] = i  # Loop index, not MIDI track!

# CORRECT: Uses actual MIDI track_id  
for i, (track_id, track) in enumerate(valid_tracks):
    occurrence['track_id'] = track_id  # Actual MIDI track
    occurrence['gm_program'] = track_programs[track_id]  # GM program
```

---

## How I Overcame Them

| Problem | Solution Approach | Time to Fix |
|---------|------------------|-------------|
| GPU Memory Overflow | Chunked processing, incremental error computation | ~4 hours |
| FISTA Gradient Error | Systematic debugging, sign correction | ~2 hours |
| Low GPU Utilization | 3-level optimization (reuse, dispatch, batching) | ~2 hours |
| FAISS Version Mismatch | Research, conda environment setup | ~3 hours |
| Generation Quality | Deep research into compression-based generation, PPM* implementation | ~2 weeks |
| Track Mapping | Careful code review, adding gm_program storage | ~2 hours |

---

## Code and Resources Used from Others

### Libraries and Frameworks

| Library | Purpose | Link |
|---------|---------|------|
| **PyTorch** | GPU tensor operations, neural network primitives | https://pytorch.org/ |
| **mido** | MIDI file parsing and manipulation | https://mido.readthedocs.io/ |
| **FAISS** | Efficient similarity search | https://github.com/facebookresearch/faiss |
| **NumPy** | Numerical operations | https://numpy.org/ |

### Algorithm References

| Algorithm | Source | Link |
|-----------|--------|------|
| **Re-Pair** | Larsson & Moffat (1999) | https://ieeexplore.ieee.org/document/755679 |
| **PPM* (Prediction by Partial Matching)** | Cleary & Witten (1984) | https://dl.acm.org/doi/10.1145/358027.358036 |
| **FISTA** | Beck & Teboulle (2009) | https://epubs.siam.org/doi/10.1137/080716542 |
| **Krichevsky-Trofimov Smoothing** | Krichevsky & Trofimov (1981) | Used for PPM* probability estimation |
| **MDL (Minimum Description Length)** | Rissanen (1978) | https://www.sciencedirect.com/science/article/pii/0005109878900055 |
| **David Lewin's GMIT** | Lewin (1987) | Book: "Generalized Musical Intervals and Transformations" |

### Research Papers Consulted

| Paper | Relevance | Link |
|-------|-----------|------|
| **LZMidi** (2025) | Compression-based symbolic music generation | https://www.researchgate.net/publication/390142695 |
| **BPE for Symbolic Music** (Fradet 2023) | Tokenization comparison | https://github.com/Natooz/BPE-Symbolic-Music |
| **MusicVAE** | Hierarchical music VAE | https://arxiv.org/abs/1803.05428 |
| **NotaGen** | Transformer-based music generation | Comparison baseline |

### Code Patterns Adapted

**GPU Memory Management Pattern** (from PyTorch forums):
```python
# Source: PyTorch Discussion Forums
# https://discuss.pytorch.org/t/how-to-clear-gpu-memory-after-training/65892

torch.cuda.empty_cache()
del large_tensor
gc.collect()
```

**IVF Index Pattern** (from FAISS Wiki):
```python
# Source: FAISS GitHub Wiki
# https://github.com/facebookresearch/faiss/wiki/Getting-started

nlist = 100
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(data)
index.nprobe = 10
```

---

## AI Assistant (Claude) Interaction Documentation

### How Claude Was Used

Throughout this project, I used Claude (Anthropic's AI assistant) for:

1. **Architecture Design Consultation** - Discussing Lewinian theory implementation
2. **Debugging Assistance** - Identifying root causes of errors
3. **Algorithm Selection** - Comparing FISTA vs Greedy vs True MDL
4. **Research Synthesis** - Understanding related work (LZMidi, BPE-Music, etc.)
5. **Code Review** - Identifying inefficiencies in GPU utilization
6. **Paper Writing** - Drafting LaTeX documentation

### Key Learning Moments from AI Interaction

**1. Understanding Re-Pair vs Generation Gap**

Claude helped me understand why perfect compression didn't lead to good generation:
> "Re-Pair creates a deterministic Straight-Line Program suitable for compression but not generation. Generation requires probabilistic sampling over alternatives — your system captures 'what patterns exist' but not 'when to use which pattern given context.'"

**My Understanding:** Compression algorithms find optimal representations for existing data, but generation needs probability distributions over possible continuations. This is why I needed to add PPM* on top of Re-Pair.

**2. GPU Memory Broadcasting Issue**

Claude identified the N×N tensor explosion:
> "Broadcasting `(N, 1, L, F)` with `(1, N, L, F)` creates `(N, N, L, F)` — for N=6254 objects, that's 512 GB!"

**My Understanding:** Implicit tensor broadcasting in PyTorch can create unexpectedly large intermediate tensors. The solution is explicit chunking with immediate cleanup.

**3. FISTA Gradient Sign**

After hours of debugging astronomical errors, Claude pointed out:
> "The gradient computation had an incorrect negative sign, causing gradient ascent instead of descent."

**My Understanding:** In optimization algorithms, a single sign error can cause the algorithm to maximize rather than minimize the objective, leading to diverging errors.

**4. Philosophy Alignment Check**

Claude helped verify my implementation matched Lewinian theory:
> "Version v41 correctly implements canonical form plus transform architecture — patterns exist in pitch-class space with one canonical instantiation, and occurrences are derived via T±n transforms. This is proper Lewinian transformational theory."

**My Understanding:** It's crucial to periodically check that implementation details align with theoretical foundations, especially when making optimizations.

1. GPU Memory Debugging Session (Nov 23, 2025)
2. FISTA Algorithm Verification (Nov 25, 2025)
3. Generation Architecture Research (Dec 7-10, 2025)
4. PPM* Implementation Guidance (Dec 10, 2025)

---

## Version History and Milestones

| Version | Date | Key Changes | Status |
|---------|------|-------------|--------|
| v6-v20 | Nov 2025 | Foundation, basic primitives | ✅ Complete |
| v21-v30 | Nov 2025 | GPU optimization, memory fixes | ✅ Complete |
| v32 | Nov 23 | SEQUITUR + D24, 42 transforms | ✅ Complete |
| v41 | Nov 30 | Canonical form, 99.9% derivation | ✅ Complete |
| v43 | Dec 1 | Complete primitive implementation | ✅ Complete |
| v53 | Dec 7 | 10K patterns, 100% reconstruction | ✅ Complete |
| v54 | Dec 10 | Per-instrument vocabularies, PPM* | 🔧 In Progress |

---

## v53 Compression Stats

| Metric                 | Value                              |
|------------------------|-------------------------------------|
| **Notes**              | 4,273,245                           |
| **Patterns stored**    | 10,000                              |
| **Encoded tokens**     | ~1,607,000 (4.27M / 2.66)           |
| **Grammar storage**    | 10,000 × ~50 bytes ≈ **500 KB**     |
| **Encoded storage**    | 1,607,000 × 2 bytes ≈ **3.2 MB**    |
| **Total storage**      | **~3.7 MB**                         |
| **Original size**      | 4,273,245 × 8 bytes ≈ **34 MB**     |
| **Compression ratio**  | **~9.2×**                           |



## Running the Code

### Prerequisites
```bash
# Python 3.10+
# CUDA 12.x compatible GPU (tested on A100 40GB)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mido numpy faiss-gpu tqdm
```

### Basic Usage
```bash
# Pattern discovery
python scripts/run_pure_contour_pipeline.py /path/to/midi_corpus \
    --output checkpoint_v54.npz \
    --max-files 1000 \
    --max-rules 10000 \


# Generation (after training)
python scripts/generate_music.py checkpoint_v54.npz \
    --output generated.mid \
    --style "big_band"
```

---

## Future Work

1. **Scale to 10,000+ files** across multiple genres
2. **User study** with musicologists rating pattern quality
3. **Style transfer experiments** (Jazz → Classical)
4. **Cover song detection** application
5. **Real-time generation** interface

---

## Acknowledgments

- David Lewin for Generalized Musical Intervals and Transformations theory
- Anthropic's Claude for extensive debugging and architecture guidance
- The FAISS team at Facebook Research for similarity search infrastructure
- David Cope's EMI, a similar philosophy and system
- PyTorch developers for GPU tensor operations

---

## References

1. Lewin, D. (1987). *Generalized Musical Intervals and Transformations*. Yale University Press.
2. Rissanen, J. (1978). "Modeling by shortest data description." *Automatica*, 14(5), 465-471.
3. Larsson, N. J., & Moffat, A. (1999). "Offline dictionary-based compression." *DCC*, 296-305.
4. Beck, A., & Teboulle, M. (2009). "A fast iterative shrinkage-thresholding algorithm." *SIAM J. Imaging Sciences*, 2(1), 183-202.
5. Cleary, J., & Witten, I. (1984). "Data compression using adaptive coding and partial string matching." *IEEE Trans. Communications*, 32(4), 396-402.

---

*Last Updated: December 11, 2025*
