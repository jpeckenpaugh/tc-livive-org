# Concept Document: Browser-Based "Training vs. Inference" Learning Lab

## 1. Goal & Audience
* **Goal:** Create an interactive, inspectable web app hosted on GitHub Pages demonstrating the concrete difference between **Untrained Inference**, **Training (Learning)**, and **Trained Inference**.
* **Audience:** A sharp business professional with no CS background who is currently learning core hardware abstractions (RAM vs. Storage, CPU vs. GPU) and wants to demystify "how AI actually works."
* **Core Philosophy:** No magic black boxes. Every calculation, tally mark, weight, and bias must be rendered directly in the DOM as inspectable, human-readable numbers.

---

## 2. Technical Stack & Constraints
* **Language & Runtime:** Single-Page Application (SPA) built with **Vanilla HTML, CSS, and Modern JavaScript** (ES6+).
* **Dependencies:** **Zero external libraries.** No npm, no React, no TensorFlow.js, no Pyodide.
* **Hosting Target:** Static hosting via GitHub Pages (`index.html`).
* **Design Pattern:** Clean component or view separation using modern CSS and modal/overlay workflows for training runs.

---

## 3. Problem Domain: Text Sentiment Classification
Instead of heavy image models, the demo uses short business-style text snippets (customer reviews/feedback) classified as **Positive (1)** or **Negative (0)**.

### Fixed Training Dataset (10 Positive, 10 Negative)
```javascript
const TRAINING_DATA = [
  // Positive (1)
  { text: "great service and friendly staff", label: 1 },
  { text: "awesome product loved the experience", label: 1 },
  { text: "quick delivery and excellent quality", label: 1 },
  { text: "very helpful support team", label: 1 },
  { text: "loved the clean design and speed", label: 1 },
  { text: "super happy with this purchase", label: 1 },
  { text: "fantastic performance and reliable", label: 1 },
  { text: "easy to use and highly recommended", label: 1 },
  { text: "impressive results in record time", label: 1 },
  { text: "wonderful experience will buy again", label: 1 },

  // Negative (0)
  { text: "terrible experience rude customer support", label: 0 },
  { text: "awful quality broke on day one", label: 0 },
  { text: "horrible service very slow delivery", label: 0 },
  { text: "poor build and disappointing waste", label: 0 },
  { text: "waste of money completely broken", label: 0 },
  { text: "bad customer service very unhelpful", label: 0 },
  { text: "worst product ever completely useless", label: 0 },
  { text: "annoying glitches and constant errors", label: 0 },
  { text: "slow and frustrating will not return", label: 0 },
  { text: "disappointing outcome highly frustrated", label: 0 }
];
```

---

## 4. The Pedagogical Ladder ("Crawl, Walk, Jog")

The SPA should support 3 selectable stages:

### Stage 1: Crawl — Tally Marks (Naive Bayes intuition)
* **Concept:** Why data matters. Words are simply counted in two buckets.
* **Inspectable State:** A live table showing:
  `{ word: "great", posCount: 7, negCount: 0 }`
  `{ word: "broken", posCount: 0, negCount: 5 }`
* **Inference:** Compares positive word hits vs. negative word hits in the test sentence.
* **Key Lesson:** Machine learning starts with organized observations.

### Stage 2: Walk — Dials & The Training Loop (Logistic Regression via SGD)
* **Concept:** Introducing "weights" (point values) and the iterative training loop.
* **Initial State (Untrained):** All word weights are set to `0.0` and `bias = 0.0`.
  * *Untrained Inference:* Any sentence scores exactly $50\%$ (neutral coin toss).
* **The Training Loop (Inside Modal):**
  * Supports a **"Step Once"** button and a **"Run All Epochs"** button.
  * For each sample:
    1. **Predict:** $z = \text{bias} + \sum (\text{weight}_i \cdot \text{word\_present}_i)$
    2. **Activate:** $P = \frac{1}{1 + e^{-z}}$ (Sigmoid)
    3. **Calculate Error:** $\text{error} = \text{label} - P$
    4. **Update Dials:** $\text{weight}_i \leftarrow \text{weight}_i + (\alpha \cdot \text{error} \cdot \text{word\_present}_i)$
* **Inspectable State:** A dynamic table showing the weights update in real time with subtle color flashes (green for positive shifts, red for negative).

### Stage 3: Jog — The Single-Neuron Perceptron (Neural Network Foundation)
* **Concept:** Demystify the buzzword. Show that the artificial neuron is the exact same mathematical engine as Stage 2.
* **Visual Representation:** An SVG/Canvas diagram of a single neuron:
  * Inputs ($x_1, x_2, \dots$) feeding along lines labeled with current weights ($w_1, w_2, \dots$).
  * A central summation node ($\sum w_i x_i + b$).
  * An activation gate (Sigmoid curve).
  * A single output probability meter ($0.0 \to 1.0$).
* **Key Lesson:** Modern AI architectures are simply large, layered networks of these individual units.

---

## 5. UI/UX Layout & User Journey

### Top Header & Mode Switcher
* Segmented control toggle: **[ Stage 1: Crawl ] | [ Stage 2: Walk ] | [ Stage 3: Jog ]**
* Brief 1-sentence descriptor explaining the active stage's mental model.

### Main Workspace (Split View)
* **Left Panel: The Model's Brain (Inspectable State)**
  * Stage 1: Word Count Frequency Table.
  * Stage 2: Word Weights & Bias Table (`word`, `weight value`, `direction indicator`).
  * Stage 3: Graphical Neuron diagram with real-time value callouts.
* **Right Panel: The Interactive Playground**
  * **Section A: Test Playground (Inference Mode)**
    * Text input box with sample sentence buttons (e.g., *"great delivery fast service"*, *"terrible product broke immediately"*).
    * Large gauge / meter: **Sentiment Score (% Positive / % Negative)**.
    * Real-time calculation breakdown: Show step-by-step arithmetic ($z = w_1 + w_2 + \dots$).
  * **Section B: The Training Control Center**
    * Display status: `Model Status: [ Untrained / Training / Trained ]`.
    * Button: **"Open Training Session"** (launches modal).

### Training Modal / Overlay
* **Visual Feedback:**
  * Epoch counter (e.g., `Epoch 5 / 20`).
  * Real-time Loss meter (showing error dropping toward zero).
  * Current sentence being evaluated and the instantaneous adjustment made to the weights.
* **Controls:**
  * `[ Step 1 Example ]` (for slow, pedagogical walkthrough).
  * `[ Run to Completion ]` (smooth 2–3 second automated loop with subtle UI ticks).
  * `[ Reset Weights to Zero ]` (allows repeating the "Untrained vs. Trained" comparison).

---

## 6. Verification & Acceptance Criteria for POC
1. **Fresh Page Load:** Running inference on any sentence yields exactly $50\%$ probability in Stage 2/3 (proving uncalibrated state).
2. **Post-Training Inference:** Running positive test sentences scores $>85\%$; running negative test sentences scores $<15\%$.
3. **No External Network Calls:** The app runs entirely offline once loaded; inspectable in browser dev tools without console errors.
4. **Interactive Weight Inspection:** The user can sort the weights table to clearly see words like `"great"` rise to $+1.5$ and `"terrible"` fall to $-1.5$.