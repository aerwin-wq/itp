# Midterm — Phases 1–4

This folder contains my midterm submission files.

## Files
- **Phase1.jpg** — original sketch (Phase 1).
- **Phase2.js** — direct p5.js translation on a 150×150 canvas with more than three shapes.
- **Phase3.js** — adds `drawObject(x, y, s)` using `translate()` and `scale()`; called twice; uses `push()`/`pop()`.
- **Phase4.js** — tiles the canvas using nested loops; scales each object to maximize size in each grid cell without resizing the window.

### Notes
- The **natural design space** for the object is 150×150; Phase 3 and 4 keep this as the baseline and scale relative to it.
- In **Phase 4**, change `COLS` and `ROWS` to try 5×5, 10×10, or 20×20. The canvas stays the same size; the object scales per cell.
