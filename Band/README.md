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

## 📁 File Structure

```
aerwin-wq.github.io/
├── _config.yml                 # Jekyll configuration
├── _layouts/
│   └── default.html           # Custom Jekyll layout
├── assets/
│   ├── css/
│   │   └── style.scss         # Main stylesheet (Sass)
│   └── js/
│       └── parallax.js        # Custom parallax engine
├── index.md                   # Main content (Markdown + HTML)
├── qav24tl.gif               # Fried background image
├── rowleytransparent.gif     # Parallax layer image
├── i-said-hey-he-man.gif     # Header image
└── README.md                 # This file
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

## 👥 Credits

**Band**: ITPP (Isaac, Ryan, Arlo)
**Developer**: Built with Claude Code
**Fonts**: Google Fonts (Rubik Maze)
**Hosting**: GitHub Pages

---

**Last Updated**: November 2025
**Built with**: ❤️ and way too many CSS animations
