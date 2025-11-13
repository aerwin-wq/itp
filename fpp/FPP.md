
# **Harmony Atlas — Final Project Proposal**

## **Title**
**Harmony Atlas: A Searchable Database of Chord Progressions and Harmony Types**

## **1-Sentence Overview**
A web-based database that organizes musical excerpts by harmony type (modal interchange, secondary dominants, chromatic mediants, etc.), using MuseScore scores and Basic Pitch–generated MIDI. The frontend is built in JavaScript, the backend in Python, and the full project is hosted on Hugging Face Spaces.

---

## **Resources (3 Hyperlinked Sources)**  
- **MuseScore Score Library** — https://musescore.com/sheetmusic  
- **Basic Pitch (Spotify)** — https://basicpitch.spotify.com/  
- **Open Studio Music Analysis Channel** — https://www.youtube.com/openstudiojazz  

---

## **Project Overview**
*Harmony Atlas* is a small but functional web platform that allows users to browse musical excerpts based on their harmonic function, rather than composer or genre. The project consists of three major components:

### **Frontend — JavaScript UI (Hugging Face Spaces — Static Space)**
- Designed using HTML, CSS, and JavaScript.  
- Provides:
  - A searchable list of excerpts  
  - Dropdown filter by harmony type  
  - Score image viewer  
  - Metadata display  
  - Optional MIDI/audio playback  
- Uses `fetch()` to request excerpt metadata from the Python backend API.
- Clean, responsive UI using simple CSS and JS DOM rendering.

Example fetch:
```js
const data = await fetch('https://harmony-backend.hf.space/api/excerpts')
  .then(res => res.json());
```

### **Backend — Python API (Hugging Face Spaces — Python Space)**
The backend handles all data organization and exposes the dataset through REST endpoints. It includes:

A Python (FastAPI or Flask) server with routes like:
- `/api/excerpts`
- `/api/excerpt/<id>`
- `/api/search?harmony_type=secondary dominant`

A structured dataset containing:
- `scores/` — PNG/PDF score snippets
- `midi/` — short MIDI files
- `excerpts.json` — metadata for every entry

Backend logic written entirely in Python:
- Basic Pitch transcription
- MIDI file cleanup
- Data parsing, formatting, and storage

**Basic Pitch Integration (Python):**
```python
from basic_pitch.inference import predict

predict(
    audio_file="input.wav",
    output_directory="output_midi/"
)
```

All preprocessing (downloading scores, transcribing audio, slicing excerpts, tagging harmony types) is done via Python scripts before populating the database.

### **Dataset Schema**
Every entry in the database follows this structure:

```json
{
  "id": "unique_id",
  "title": "Prelude in E Minor",
  "composer": "Chopin",
  "source": "MuseScore",
  "harmony_type": ["modal interchange", "borrowed iv chord"],
  "score_image": "scores/prelude_em.png",
  "midi": "midi/prelude_em.mid",
  "notes": "Borrowed iv chord in measure 4."
}
```

### **Technical Hosting Setup — Hugging Face Spaces**
The project will use two Spaces:

#### 1. Frontend Space (Static)
Contains:
- `index.html`
- `style.css`
- `app.js`
- Pure client-side rendering using JS.

#### 2. Backend Space (Python/Server app)
Contains:
- `app.py`
- dataset folders (`/scores`, `/midi`, `/json`)
- Basic Pitch installed in the environment
- Public API consumed by the frontend Space.

The frontend communicates with the backend using full HF URLs like:
```
https://harmony-atlas-backend.hf.space/api/excerpts
```

---

## **Overlap With Other Work**
The harmonic knowledge overlaps with general music coursework, but no part of this project is being submitted for any other class. This class is taught by Professor Rachel.

---

## **Outcomes**

### **GOOD Outcome (Minimum Guaranteed)**
- Working frontend + backend
- At least 50 labeled excerpts
- Functional search and harmony-type filtering

### **BETTER Outcome (Expected Goal)**
- 100–200 excerpts
- Score viewer + MIDI playback
- Responsive, polished UI
- Clean metadata formatting

### **BEST Outcome (Ideal Hope)**
- 500+ tagged excerpts
- Advanced search and filtering
- Optional simple chord-suggestion model using Basic Pitch MIDI + rule-based detection
- Fully polished UX with smooth animations and zoomable score viewer

---

## **Project Timeline (Calendar With Dates)**

### **Week 1 — Nov 15–Nov 21**
- Finalize harmony categories
- Build dataset schema (JSON/CSV)
- Test downloading MuseScore scores
- Test Python Basic Pitch transcription
- Begin folder structure setup

### **Week 2 — Nov 22–Nov 28**
- Extract audio → MIDI using Python + Basic Pitch
- Cut score images and assign harmony types
- Populate initial dataset items
- Write backend preprocessing scripts
- Begin labeling 20–40 excerpts

### **Week 3 — Nov 29–Dec 5**
- Build JavaScript frontend:
  - search bar
  - category filter
  - excerpt grid layout
- Deploy prototype backend API on HF Spaces
- Display first 10–20 items in UI

### **Week 4 — Dec 6–Dec 12**
- Expand database with more examples
- Improve UI styling
- Integrate MIDI playback
- Add full metadata support

### **Final Week — Dec 13–Dec 18**
- Final dataset polish
- Debug all UI/API functions
- Add last entries (GOOD/BETTER/BEST outcome)
- Finalize README + submit project
