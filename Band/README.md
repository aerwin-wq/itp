# ITPP Band Website

https://aerwin-wq.github.io/

A psychedelic, parallax-heavy website for the jazz fusion trio ITPP, hosted on GitHub Pages with Jekyll.



## 🚀 Hosting & Deployment

- **Platform**: GitHub Pages
- **Static Site Generator**: Jekyll
- **Theme Base**: Minimal (heavily customized)
- **Deployment**: Automatic via GitHub Actions on push to `main` branch

## 🎨 Design Philosophy

This website embraces controlled chaos with multiple layers of parallax effects, animated gradients, oscillating sections, and a heavily distorted background. Each section has unique typography and color schemes to create a distinct visual identity.

## 🏗️ Architecture

### Layout Structure
- **Custom Jekyll Layout**: `_layouts/default.html`
  - Minimal wrapper - content renders directly in `<body>`
  - No traditional header/footer from theme
  - Google Fonts preconnect for performance
  - Rubik Maze font for distinctive headers

### Content Structure
- **Single Page Application**: `index.md`
  - Fixed navigation bar
  - 5 main sections: Home, About, Reviews, Projects, Contact
  - Each section wrapped in `.content-section` divs with unique classes

## 🎭 Parallax Effects

### Custom JavaScript Parallax Engine
**File**: `assets/js/parallax.js`

No external parallax libraries used - all effects are custom-built using vanilla JavaScript and requestAnimationFrame for 60fps performance.

#### Rowley Background Layers (50 instances)
- **Position**: Fixed background container at z-index: -1
- **Movement Types**:
  - **Continuous Floating**: Sine wave motion using `Math.sin()` and `Math.cos()`
    - X-axis: 30px amplitude
    - Y-axis: 20px amplitude
    - Random frequency per layer (0.3 to 1.0)
    - Random phase offsets for varied starting positions
  - **Scroll Parallax**: Moves at different speeds based on `data-speed` attribute (0.2x to 0.9x)
  - **Mouse Parallax**: Follows cursor movement (50px range scaled by speed)
  - **Y-Axis Mirroring**: 50% of layers randomly flipped using `scaleY(-1)`
- **Implementation**:
  ```javascript
  const floatX = Math.sin(time * frequency + phaseX) * 30 * speed;
  const floatY = Math.cos(time * frequency * 0.8 + phaseY) * 20 * speed;
  layer.style.transform = `translate(${totalX}px, ${totalY}px) scaleY(-1)`;
  ```

### Content Section Animations
**Type**: Pure CSS animations (not JavaScript parallax)

Each section oscillates horizontally at different speeds:
- **Home**: 4s cycle, -5% to +5%
- **About**: 3.5s cycle reversed, 8% to -8%
- **Reviews**: 5s cycle (slowest), -12% to +12%
- **Projects**: 3s cycle reversed (fastest), 6% to -6%
- **Contact**: 4.5s cycle, -7% to +7%

**CSS Implementation**:
```css
.section-home {
  animation: drift-home 4s ease-in-out infinite;
}
@keyframes drift-home {
  0%, 100% { transform: translateX(-5%); }
  50% { transform: translateX(5%); }
}
```

## 🎨 CSS Styling & Effects

### Typography System

Each section uses a different font family for unique personality:

| Section | Body Font | Header Font | Purpose |
|---------|-----------|-------------|---------|
| **Home** | Rubik Maze | Rubik Maze | Maze-pattern display font |
| **About** | Courier New | Impact, Georgia | Typewriter + bold impact |
| **Reviews** | Trebuchet MS | Comic Sans MS | Casual, friendly |
| **Projects** | Verdana | Arial Black, Lucida Console | Technical, modern |
| **Contact** | Palatino | Brush Script MT | Elegant, handwritten |

### Animated Rainbow Gradients

**Headers (h1, h2)**: Fast-moving gradients using `background-clip: text`

**h1 Implementation**:
```css
h1 {
  background: linear-gradient(90deg,
    #ff0000, #ff7f00, #ffff00, #00ff00,
    #0000ff, #4b0082, #9400d3, #ff00ff,
    #ff0000);
  background-size: 400% 100%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: rainbow-gradient 2s linear infinite;
}
```

- **h1**: Full rainbow, 2s cycle, left-to-right
- **h2**: Alternate colors, 1.5s cycle, right-to-left (reversed)

### "Fried" Background Effect

**Element**: `body::before` pseudo-element
**Image**: `qav24tl.gif`

**CSS Filters (Animated)**:
```css
filter:
  saturate(500-600%)      /* Extreme color saturation */
  contrast(200-250%)      /* High contrast */
  hue-rotate(90-270deg)   /* Color shifting */
  brightness(150-180%)    /* Blown out highlights */
  blur(2-3px)             /* Slight blur */
  sepia(30-50%);          /* Vintage burn effect */
```

**Animation**: 3s cycle with three keyframes creating continuous distortion

### Section Styling

All sections share:
- **Background**: `rgba(0, 0, 0, 0.95)` - Near-black with transparency
- **Text Color**: White (`#fff`)
- **Max Width**: 600px (constrained for readability)
- **Padding**: 120px top (navbar clearance), 40px sides
- **Border Radius**: 15px
- **Box Shadow**: `0 10px 30px rgba(0, 0, 0, 0.2)`

**Unique Section Colors**:
- **Home**: White text on black
- **About**: Orange accents (#FFB366, #FFD699)
- **Reviews**: Bright green accents (#00FF66)
- **Projects**: Purple accents (#BB88FF, #CC99FF)
- **Contact**: Gold accents (#FFD700)

### Review Cards

**Layout**: CSS Grid with 20px gap
**Style**: Gradient background cards

```css
.review {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.review:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
}
```

**Star Ratings**: Golden stars (#FFD700) with text-shadow

## 🧭 Navigation

### Fixed Navbar
- **Position**: Fixed top, full width
- **Background**: `rgba(0, 0, 0, 0.85)` with `backdrop-filter: blur(10px)`
- **Z-Index**: 1000 (always on top)
- **Style**: Black transparent with white text
- **Hover Effects**:
  - Color change to `#00ff88` (bright green)
  - Vertical lift with `translateY(-2px)`
  - Animated underline that grows from 0 to 100% width

### Smooth Scrolling
JavaScript-powered smooth scroll using:
```javascript
targetSection.scrollIntoView({
  behavior: 'smooth',
  block: 'start'
});
```

## 🎭 Key Technologies

### Core Stack
- **Jekyll 3.10.0**: Static site generator
- **GitHub Pages**: Free hosting with automatic deployment
- **Sass/SCSS**: CSS preprocessing
- **Vanilla JavaScript**: No frameworks, pure ES6+

### CSS Features Used
- CSS Grid (review cards)
- Flexbox (navigation, review headers)
- CSS Animations (`@keyframes`)
- CSS Transforms (`translate`, `scaleY`)
- CSS Filters (background effects)
- Background Clip (text gradients)
- Backdrop Filter (navbar blur)
- CSS Variables (via Sass)

### JavaScript APIs Used
- `requestAnimationFrame`: Smooth 60fps animations
- `querySelector/querySelectorAll`: DOM selection
- `getBoundingClientRect`: Section positioning
- `scrollIntoView`: Smooth navigation
- `Math.sin/cos`: Sine wave animations
- `Math.random`: Randomization effects

## 🎨 Color Palette

### Primary Colors
- **Black**: `#000000` (backgrounds)
- **White**: `#ffffff` (text)
- **Bright Green**: `#00ff88` (nav hover)

### Section Accent Colors
- **Orange Spectrum**: `#FFB366`, `#FFD699` (About)
- **Green Spectrum**: `#00FF66` (Reviews)
- **Purple Spectrum**: `#BB88FF`, `#CC99FF` (Projects)
- **Gold Spectrum**: `#FFD700`, `#DAA520` (Contact)

### Rainbow Gradient
- Red → Orange → Yellow → Green → Blue → Indigo → Violet → Magenta

## 🔧 Performance Optimizations

1. **Font Preconnect**: Google Fonts preconnect for faster loading
2. **requestAnimationFrame**: Optimized animation loop
3. **will-change**: CSS hint for transform optimizations
4. **pointer-events: none**: Parallax layers don't interfere with clicks
5. **CSS-only animations**: Offloaded section movement from JavaScript
6. **Debouncing**: Smooth scroll events don't stack

## 📱 Responsive Design

**Breakpoint**: 768px

Mobile adjustments:
- Navbar font size reduced (16px → 12px)
- Navbar gap reduced (40px → 15px)
- Section transforms disabled (all set to 0%)
- Padding reduced (120px → 100px top)

## 🎯 Browser Compatibility

**Modern browsers only** (2020+):
- Chrome/Edge 88+
- Firefox 85+
- Safari 14+

**Required Features**:
- CSS `backdrop-filter`
- CSS `background-clip: text`
- ES6+ JavaScript
- requestAnimationFrame
- CSS Grid

## 🚀 Local Development

```bash
# Clone repository
git clone https://github.com/aerwin-wq/aerwin-wq.github.io.git
cd aerwin-wq.github.io

# Install Jekyll (if not already installed)
gem install jekyll bundler

# Serve locally
bundle exec jekyll serve

# View at http://localhost:4000
```

## 📝 Content Management

### Adding a New Section
1. Add section to `index.md`:
   ```html
   <div id="new-section" class="content-section section-new" data-parallax="0.5">
     <h2>New Section</h2>
     <p>Content here...</p>
   </div>
   ```

2. Add navigation link:
   ```html
   <a href="#new-section">NEW SECTION</a>
   ```

3. Add styles in `assets/css/style.scss`:
   ```scss
   .section-new {
     background: rgba(0, 0, 0, 0.95);
     transform: translateX(10%);
     font-family: 'Your Font', sans-serif;
     animation: drift-new 4s ease-in-out infinite;
   }
   ```

## 🎨 Customization Guide

### Change Parallax Speed
Edit `data-speed` attribute in parallax layers (0.1 = slow, 0.9 = fast)

### Change Section Colors
Edit section-specific styles in `style.scss`:
```scss
.section-about h2 {
  color: #YourColor;
}
```

### Change Animation Speed
Adjust animation duration in keyframes:
```scss
animation: drift-home 4s ease-in-out infinite; /* Change 4s */
```

### Add More Parallax Layers
Add to `index.md` in parallax-container:
```html
<div class="parallax-layer"
     data-speed="0.5"
     data-position="50,50"
     data-size="200"></div>
```

## 🐛 Known Issues

1. **Performance**: 50 parallax layers may lag on older devices
2. **Mobile**: Some parallax effects disabled on mobile for performance
3. **Safari**: Backdrop-filter may have reduced blur on older versions

## 📜 License

This project is open source and available for educational purposes.

## 👥 Collaboration & Credits

### Team Contributions

This website was a collaborative effort between all band members, combining creative vision with technical implementation.

#### Content & Creative Direction
**Isaac, Ryan, and Arlo** contributed:
- **Band Biographies**: Wrote individual member bios highlighting their musical backgrounds and influences
- **Text Content**: Created all written content including band description, project descriptions, and contact information
- **Asset Selection**: Chose and curated all images and GIFs used throughout the site
  - Selected the background GIF (`qav24tl.gif`)
  - Chose parallax layer GIFs (`rowleytransparent.gif`, `i-said-hey-he-man.gif`)
  - Provided creative direction on visual aesthetic

#### Technical Implementation
**Developer** (me) handled:
- **Site Architecture**: Built Jekyll structure and page layout
- **CSS Styling**: Implemented all visual effects, animations, and responsive design
  - Created "fried" background filters
  - Designed rainbow gradient headers
  - Styled review cards and section layouts
  - Applied unique typography per section
- **JavaScript Development**: Built custom parallax engine from scratch
  - Implemented 50-layer parallax system with sine wave motion
  - Added mouse-follow and scroll-based parallax
  - Created smooth scroll navigation
- **Deployment**: Set up GitHub Pages hosting and Jekyll configuration
- **Optimization**: Performance tuning and browser compatibility

### Division of Labor

| Area | Contributors | Responsibilities |
|------|--------------|------------------|
| **Content Writing** | Isaac, Ryan, Arlo | Bios, descriptions, project info |
| **Visual Assets** | Isaac, Ryan, Arlo | Image selection, GIF curation |
| **Design Direction** | Isaac, Ryan, Arlo | Color schemes, overall aesthetic |
| **HTML/CSS** | Developer | Structure, styling, animations |
| **JavaScript** | Developer | Parallax effects, interactivity |
| **Deployment** | Developer | Hosting, configuration, optimization |

This collaborative approach allowed us to create a website that authentically represents ITPP's creative vision while utilizing advanced web technologies for maximum visual impact.

---

### Credits

**Band**: ITPP (Isaac, Ryan, Arlo)
**Developer**: Built with Claude Code
**Fonts**: Google Fonts (Rubik Maze)
**Hosting**: GitHub Pages

---

## 🤖 Development Process - ChatGPT Assistance

This project was built with assistance from ChatGPT to learn and implement advanced CSS and JavaScript techniques. Below are some of the key prompts and responses that helped shape this website.

### Prompt 1: Custom Parallax Library

**Me:**
```
How do I create a custom parallax effect in JavaScript without using external libraries?
I want multiple layers moving at different speeds based on scroll position.
```

**ChatGPT:**
```
You can create a custom parallax effect using vanilla JavaScript and the requestAnimationFrame
API for smooth 60fps performance. Here's a basic implementation:

1. Select all parallax elements:
   const layers = document.querySelectorAll('.parallax-layer');

2. Listen for scroll events and calculate position:
   window.addEventListener('scroll', () => {
     const scrolled = window.pageYOffset;
     layers.forEach(layer => {
       const speed = layer.getAttribute('data-speed');
       const yPos = -(scrolled * speed);
       layer.style.transform = `translateY(${yPos}px)`;
     });
   });

3. For optimization, use requestAnimationFrame:
   let ticking = false;
   window.addEventListener('scroll', () => {
     if (!ticking) {
       window.requestAnimationFrame(updateParallax);
       ticking = true;
     }
   });

This gives you full control over the parallax behavior without dependencies.
```

### Prompt 2: Advanced Transforms

**Me:**
```
How can I make elements move in a sine wave pattern continuously,
and also flip some of them randomly on the y-axis?
```

**ChatGPT:**
```
You can use Math.sin() and Math.cos() for sine wave motion, combined with CSS transforms:

1. Create a continuous animation loop:
   let time = 0;
   function animate() {
     time += 0.01;
     elements.forEach(el => {
       const frequency = parseFloat(el.dataset.frequency);
       const phase = parseFloat(el.dataset.phase);

       // Sine wave motion
       const floatX = Math.sin(time * frequency + phase) * 30;
       const floatY = Math.cos(time * frequency * 0.8 + phase) * 20;

       // Random flip
       const flip = el.dataset.flipped === '1' ? 'scaleY(-1)' : '';
       el.style.transform = `translate(${floatX}px, ${floatY}px) ${flip}`;
     });
     requestAnimationFrame(animate);
   }

2. Initialize with random values:
   elements.forEach(el => {
     el.dataset.phase = Math.random() * Math.PI * 2;
     el.dataset.frequency = 0.3 + Math.random() * 0.7;
     el.dataset.flipped = Math.random() > 0.5 ? '1' : '0';
   });

The scaleY(-1) flips elements vertically. The random phase and frequency make each
element move differently, creating organic motion.
```

### Prompt 3: CSS Filter Effects

**Me:**
```
I want to make a background image look "deep fried" - super saturated,
high contrast, with shifting colors. How do I do this with CSS filters?
```

**ChatGPT:**
```
You can chain multiple CSS filters together to create a "fried" effect:

filter:
  saturate(500%)      /* Extreme color boost */
  contrast(200%)      /* High contrast */
  hue-rotate(90deg)   /* Shift all colors */
  brightness(150%)    /* Blow out highlights */
  blur(2px)           /* Slight blur for grain */
  sepia(30%);         /* Add vintage burn */

For animated distortion, use keyframes:

@keyframes fry {
  0% {
    filter: saturate(500%) contrast(200%) hue-rotate(90deg);
  }
  50% {
    filter: saturate(600%) contrast(250%) hue-rotate(180deg);
  }
  100% {
    filter: saturate(550%) contrast(220%) hue-rotate(270deg);
  }
}

Apply to a pseudo-element to avoid affecting content:

body::before {
  content: '';
  position: fixed;
  background-image: url('image.gif');
  filter: saturate(500%) contrast(200%);
  animation: fry 3s infinite alternate;
}

The combination of high saturation, contrast, and rotating hues creates that
oversaturated "meme" aesthetic.
```

### Prompt 4: Animated Text Gradients

**Me:**
```
How do I make text with a moving rainbow gradient that loops infinitely?
```

**ChatGPT:**
```
Use background-clip: text to make gradients show through text:

1. Create the gradient (extend it beyond 100% for smooth looping):
   h1 {
     background: linear-gradient(90deg,
       #ff0000, #ff7f00, #ffff00, #00ff00,
       #0000ff, #4b0082, #9400d3, #ff00ff,
       #ff0000);  /* Repeat first color for seamless loop */
     background-size: 400% 100%;
   }

2. Clip background to text only:
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   background-clip: text;

3. Animate the background position:
   animation: rainbow-gradient 2s linear infinite;

   @keyframes rainbow-gradient {
     0% { background-position: 0% 50%; }
     100% { background-position: 100% 50%; }
   }

The 400% background-size allows the gradient to pan smoothly across the text.
Using linear timing creates constant motion, and infinite makes it loop forever.
```

### Prompt 5: Section Oscillation

**Me:**
```
I want different sections of my page to slide back and forth horizontally
at different speeds. How do I do this with CSS?
```

**ChatGPT:**
```
Use CSS @keyframes animations with translateX:

1. Create unique animations for each section:
   @keyframes drift-home {
     0%, 100% { transform: translateX(-5%); }
     50% { transform: translateX(5%); }
   }

   @keyframes drift-about {
     0%, 100% { transform: translateX(8%); }
     50% { transform: translateX(-8%); }
   }

2. Apply with different durations and directions:
   .section-home {
     animation: drift-home 4s ease-in-out infinite;
   }

   .section-about {
     animation: drift-about 3.5s ease-in-out infinite reverse;
   }

Key points:
- ease-in-out creates smooth acceleration/deceleration
- infinite makes it loop forever
- reverse makes some sections move opposite directions
- Different durations (3s, 4s, 5s) create chaotic, unsynchronized movement

This creates the effect of sections "breathing" at different rates.
```

### Key Takeaways

**What I learned from ChatGPT:**
- No external parallax libraries needed - vanilla JS with requestAnimationFrame is sufficient
- Math.sin/cos can create organic floating motion
- CSS filters can be chained for extreme visual effects
- background-clip: text enables gradient text
- CSS animations are more performant than JavaScript for simple movements
- Random values (Math.random()) add natural variation to effects

**Implementation approach:**
1. Started with basic parallax concepts from ChatGPT
2. Iterated on the code to add multiple effect layers
3. Combined scroll, mouse, and continuous animation
4. Optimized with requestAnimationFrame and CSS animations
5. Added random variations for organic feel

This project demonstrates that modern web effects don't require heavy libraries -
just understanding of core CSS and JavaScript APIs!

---

