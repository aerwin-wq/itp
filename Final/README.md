
# Harmony Atlas: MIDI Music Analysis and Remixing

## Project Documentation

**Author:** Arlo Erwin
**Date:** November-December 2025  
**Project Duration:** ~3 weeks of active development

---

## What I Did

I developed **Harmony Atlas**, a neurosymbolic AI system that computationally implements David Lewin's transformational music theory for MIDI analysis and generation. The core philosophy is "discovery not prescription" — using compression algorithms to discover musical relationships rather than hardcoding music theory rules.

### Project Goals
1. Achieve 97-100% lossless MIDI reconstruction
2. Discover musical patterns using <500 universal transforms
3. Create an interpretable, editable "gene editor" for music
4. Generalize learned patterns across unseen files
5. Enable style transfer and cover song detection

### Key Accomplishments

| Milestone | Result |
|-----------|--------|
| Pattern Discovery | 10,000+ patterns discovered via Re-Pair compression |
| Occurrence Tracking | 2.6 million pattern occurrences mapped |
| Reconstruction Quality | 100% lossless compression achieved |
| Compression Ratio | 3.48x on 500-file corpus |
| GPU Optimization | 11-18x speedup with 85-95% A100 utilization |
| Transform Discovery | 42 MDL-optimized transforms |
| Cross-Track Relations | 38,000+ orchestration rules discovered |

---

## How I Did It

### Theoretical Foundation

The system represents music as "DNA in transform space" using 17 irreducible mathematical primitives:

**Pitch Operations:**
- Transposition (T₁-T₁₁): Shift all pitches by n semitones
- Inversion (I₀-I₁₁): Flip pitch contour around axis
- Retrograde (R): Reverse note sequence

**Neo-Riemannian Transforms:**
- Parallel (P): Major ↔ minor exchange
- Leading-tone (L): Leading tone transformation
- Relative (R): Relative major/minor exchange

**Temporal Operations:**
- Time Scaling (τ): Tempo changes
- Rhythmic Transforms: Duration ratios

**Multitrack Operations:**
- TrackDerive: Cross-instrument relationships
- InstrumentFilter: Section isolation

### Architecture Evolution
```
Phase 1 (v6-v20): Foundation
├── Basic primitives (T, I, R)
├── Multi-scale pattern extraction (16, 32, 64, 128, 256 timesteps)
└── Initial factorization (pitch × rhythm × velocity × duration)

Phase 2 (v20-v30): GPU Optimization
├── Tensorized operations on NVIDIA A100
├── Chunked processing (8 compositions at a time)
├── Memory reduction: 42+ GB → <4 GB
└── General MIDI program encoding (vs track positions)

Phase 3 (v31-v41): Canonical Form Architecture
├── Patterns store canonical (T-normalized) pitches
├── Occurrences derive absolute pitch via octave transforms
├── Cross-track discovery ("trombone = trumpet + T(-7)")
└── 99.9% derivation rate achieved

Phase 4 (v42-v43): Primitive Completeness
├── τ, v, d transform discovery (previously only stored)
├── TrackDerive per-occurrence (not just summary stats)
├── Per-piece interval magnitude discovery
└── Feature importance for emergent "keys"

Phase 5 (v53-v54): Generation Improvement
├── Re-Pair + PPM* hybrid architecture
├── 6-level probabilistic sampling system
├── Per-instrument pattern vocabularies
└── GPU-accelerated clustering for style variables
```

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

## Problems I Faced (ABDYDB: Always Be Documenting Your Debugging!)

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
| Low GPU Utilization | 3-level optimization (reuse, dispatch, batching) | ~8 hours |
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

### Screenshots of AI Interactions

*Note: As this is a markdown document, screenshots would be included as images in the final submission. Key conversations documented in this project include:*

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
    --top-instruments 15

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
