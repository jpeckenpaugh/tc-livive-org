# Concept Specification (Revision 2): Browser-Based "Training vs. Inference" Learning Lab

## 1. Executive Summary & Vision
* **Product:** Interactive, inspectable web lab hosted on GitHub Pages demonstrating how machine learning models learn dials/weights and execute predictions.
* **Target Audience:** Business professionals, leaders, and curious non-programmers seeking to demystify "How AI actually works" through transparent, inspectable systems.
* **Core Philosophy:** **Zero Black Boxes.** No hidden frameworks, no opaque matrix libraries, and no hand-waving. Every weight, calculation, and activation is rendered directly in the DOM in human-readable numbers.
* **Key Pedagogical Shift in Rev 2:** Moving away from repetitive representations of a single binary equation (Crawl/Walk/Jog) to an authentic, three-phase architectural and conceptual journey:
  1. **Phase 1: "Black Text on White Space"** — The 1-Neuron Binary Teeter-Totter (and its failure modes).
  2. **Phase 2: "Red, Green, and Blue"** — The 3-Neuron Spectrum (Positive, Negative, Neutral).
  3. **Phase 3: "Three Lanes in Two Dimensions"** — Multi-Task Enterprise Head (Sentiment + Actionable Department Routing).

---

## 2. Technical Stack & Architectural Constraints
* **Platform:** Static Single-Page Application (SPA) deployable via GitHub Pages (`index.html`).
* **Languages:** Vanilla HTML5, Modern CSS (variables, grid/flexbox, transitions), ES6+ JavaScript.
* **Dependencies:** **Zero external dependencies.** (No React, Vue, TensorFlow.js, Pyodide, or math libraries). Runs completely client-side and offline once loaded.
* **Styling & UI:** Clean dark-mode dashboard with high visual contrast, responsive layout, modal/dialog workflows for training sessions, and real-time SVG diagrams.

---

## 3. The Three-Phase Curriculum

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE 3-PHASE LEARNING JOURNEY                          │
├─────────────────────────────────────┬───────────────────────────────────────┤
│ Phase 1: Black Text on White Space  │ 1 Neuron (Sigmoid)                    │
│                                     │ Negative is merely lack of Positive   │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ Phase 2: Red, Green, and Blue       │ 3 Neurons (Pos / Neg / Neu)           │
│                                     │ Distinct sentiment channels / blend   │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ Phase 3: Three Lanes in Two Dims    │ Multi-Task Head (3x3 Matrix)          │
│                                     │ Sentiment + Routing (CS/Billing/Sales)│
└─────────────────────────────────────┴───────────────────────────────────────┘
```

---

### Phase 1: "Black Text on White Space" (The 1-Neuron Teeter-Totter)
* **Metaphor:** High-contrast, binary, 1-bit presence/absence. Either ink exists on paper or it doesn't.
* **The Mathematical Reality:**
  * Single output neuron: $z = b + \sum (w_i \cdot x_i)$
  * Sigmoid squashing: $P = \frac{1}{1 + e^{-z}}$
  * Target output: $1$ (Positive) or $0$ (Not Positive).
* **Display Toggles:**
  * **View A (Raw Probability):** Standard $0.0 \to 1.0$ with $0.5$ decision threshold.
  * **View B (Centered Polarity & Confidence):** Scale of $-1.0$ (Negative) to $+1.0$ (Positive) with $0.0$ as Neutral origin, confidence $= 2 \times |P - 0.5|$.
* **The Critical Pedagogical Lesson & Failure Mode:**
  * *Untrained State:* All weights are $0.0$, output lands at exact center ($0.5$ or $0.0$ polarity).
  * *The Trap:* Students test a factual statement (*"The package was delivered on Tuesday"*). The model outputs $50\%$ (a coin toss).
  * *The Takeaway:* A single binary neuron forces "Negative" to just mean "the absence of Positive." It cannot distinguish between **indifference/factual neutrality** and a coin toss between love and hate.

---

### Phase 2: "Red, Green, and Blue" (The 3-Neuron Spectrum)
* **Metaphor:** RGB color channels. In color display, Red is not the "absence of Green"; they are independent active channels that blend into full spectrum.
* **The Mathematical Reality:**
  * Three independent output neurons:
    * 🟢 **Green Neuron (Positive):** $z_{\text{pos}} = b_{\text{pos}} + \sum w_{i,\text{pos}} x_i \to P_{\text{pos}} = \sigma(z_{\text{pos}})$
    * 🔴 **Red Neuron (Negative):** $z_{\text{neg}} = b_{\text{neg}} + \sum w_{i,\text{neg}} x_i \to P_{\text{neg}} = \sigma(z_{\text{neg}})$
    * 🔵 **Blue Neuron (Neutral):** $z_{\text{neu}} = b_{\text{neu}} + \sum w_{i,\text{neu}} x_i \to P_{\text{neu}} = \sigma(z_{\text{neu}})$
* **Output Evaluation:**
  * Multi-meter radar view / three independent channels plus an integrated blend.
* **What This Unlocks:**
  * *"Awesome friendly service!"* $\to$ High Green (0.92), Low Red (0.05), Low Blue (0.10) $\implies$ **Pure Positive**.
  * *"Completely broken waste of money"* $\to$ Low Green (0.04), High Red (0.94), Low Blue (0.08) $\implies$ **Pure Negative**.
  * *"The package was delivered on Tuesday"* $\to$ Low Green (0.08), Low Red (0.05), High Blue (0.88) $\implies$ **True Factual Neutral**.
  * *"The food was delicious, but the service was horrific"* $\to$ High Green (0.85) AND High Red (0.88) $\implies$ **Mixed / Ambivalent (Yellow)**.
* **The Takeaway:** Human sentiment is multi-dimensional. Vocabulary specializes across specific sentiment channels.

---

### Phase 3: "Three Lanes in Two Dimensions" (The Enterprise Multi-Head Model)
* **Metaphor:** Traffic moving through a multi-lane intersection. A message is evaluated along two independent axes: **How does the customer feel?** and **Where should the company route this?**
* **The Mathematical Reality:** Multi-Task Classification using a shared vocabulary input feeding two separate multi-neuron heads:
  * **Dimension 1 (Tone Head - 3 Neurons):** Positive, Negative, Neutral.
  * **Dimension 2 (Routing Destination Head - 3 Neurons):** 
    * 🎧 **Customer Support / Service** (troubleshooting, bugs, repairs)
    * 💳 **Billing & Accounts** (invoices, refunds, unexpected charges)
    * 💼 **Sales & Upgrades** (quotes, new seats, enterprise pricing)
* **The Inspectable "Brain":**
  * Weights table presents a clean, sub-tabbed or filtered view:
    * **Tone Weights Tab** (Pos, Neg, Neu)
    * **Department Weights Tab** (Support, Billing, Sales)
    * **Word Radar Inspector** (Click any word in a test sentence to see its 6 dials light up).
  * Words like *"refund"*, *"charge"*, *"invoice"* activate the **Billing** lane without corrupting Tone.
  * Words like *"crash"*, *"broken"*, *"bug"*, *"error"* activate the **Support** lane.
  * Words like *"quote"*, *"pricing"*, *"upgrade"*, *"contract"* activate the **Sales** lane.
* **Interactive Enterprise Matrix Output:**
  * Displays a $3 \times 3$ heat grid (Tone $\times$ Department).
  * Example: *"Cancel my subscription, you double-charged my card"* lights up:
    * **Tone:** 🔴 Negative
    * **Department:** 💳 Billing
    * **Action Result:** Urgent Escalation to Billing Team.
* **The Takeaway for Business Leaders:** Real-world enterprise AI models don't just gauge opinions—they map unstructured human communication directly into actionable workflows.

---

## 4. Dataset Specification & Lexical Disentanglement

To train both the 3-sentiment head and the 3-department head, the dataset contains 30 balanced, realistic business feedback snippets.

### 4.1 Cross-Disentanglement Design Rules
1. **Domain Anchors across Multiple Tones:** Words defining departments must appear across different sentiments so the model learns that department words (e.g., *"refund"*, *"support"*, *"pricing"*) are orthogonal to tone:
   * *"prompt refund received, thank you"* $\implies$ `[pos, billing]`
   * *"still waiting for my refund, unacceptable"* $\implies$ `[neg, billing]`
   * *"where can I request a refund"* $\implies$ `[neu, billing]`
2. **Distinct Lexical Anchors for Neutral:** The Blue/Neutral neuron requires explicit positive training signals rather than just an absence of words:
   * Anchors include: *"inquiry"*, *"request"*, *"status"*, *"statement"*, *"receipt"*, *"scheduled"*, *"information"*.

### 4.2 Dataset Schema
Each sample contains:
* `text`: Clean review or ticket snippet.
* `sentiment`: `pos` | `neg` | `neu`
* `department`: `support` | `billing` | `sales`

### 4.3 Training Samples (Full 30-Item Matrix)
```javascript
const TRAINING_DATA = [
  // --- Support (10 samples: 3 Pos, 4 Neg, 3 Neu) ---
  { text: "support agent quickly resolved the bug", sentiment: "pos", department: "support" },
  { text: "great assistance fixed my login error immediately", sentiment: "pos", department: "support" },
  { text: "helpful technician resolved the technical glitch", sentiment: "pos", department: "support" },
  { text: "terrible app keeps crashing on login", sentiment: "neg", department: "support" },
  { text: "broken update caused frustrating system freeze", sentiment: "neg", department: "support" },
  { text: "awful experience customer support never answered ticket", sentiment: "neg", department: "support" },
  { text: "unresolved bug completely broke our daily workflow", sentiment: "neg", department: "support" },
  { text: "how do I reset my password on mobile", sentiment: "neu", department: "support" },
  { text: "requesting technical documentation for api configuration", sentiment: "neu", department: "support" },
  { text: "checking status of open maintenance ticket", sentiment: "neu", department: "support" },

  // --- Billing (10 samples: 3 Pos, 4 Neg, 3 Neu) ---
  { text: "prompt refund received thank you very much", sentiment: "pos", department: "billing" },
  { text: "billing team quickly corrected invoice discrepancy", sentiment: "pos", department: "billing" },
  { text: "smooth payment processing and clear receipt", sentiment: "pos", department: "billing" },
  { text: "unauthorized double charge on my credit card", sentiment: "neg", department: "billing" },
  { text: "still waiting for my refund unacceptable delay", sentiment: "neg", department: "billing" },
  { text: "terrible unexpected fee charged without notice", sentiment: "neg", department: "billing" },
  { text: "disputing incorrect invoice completely wrong total", sentiment: "neg", department: "billing" },
  { text: "where can I download my annual tax receipt", sentiment: "neu", department: "billing" },
  { text: "inquiry regarding upcoming subscription renewal date", sentiment: "neu", department: "billing" },
  { text: "requesting copy of monthly billing statement", sentiment: "neu", department: "billing" },

  // --- Sales (10 samples: 3 Pos, 4 Neg, 3 Neu) ---
  { text: "excited to upgrade our contract to enterprise tier", sentiment: "pos", department: "sales" },
  { text: "fantastic onboarding demo love the product features", sentiment: "pos", department: "sales" },
  { text: "wonderful sales representative offered great discount", sentiment: "pos", department: "sales" },
  { text: "pricing is completely unreasonable for small teams", sentiment: "neg", department: "sales" },
  { text: "refused to cancel contract pushy aggressive salesperson", sentiment: "neg", department: "sales" },
  { text: "ridiculous hidden fees in the annual proposal", sentiment: "neg", department: "sales" },
  { text: "expensive quote offers terrible value for money", sentiment: "neg", department: "sales" },
  { text: "requesting formal price quote for ten seats", sentiment: "neu", department: "sales" },
  { text: "schedule an introductory demonstration for our team", sentiment: "neu", department: "sales" },
  { text: "inquiry about enterprise licensing options and tiers", sentiment: "neu", department: "sales" }
];
```

---

## 5. Mathematical & Algorithmic Formulation

### 5.1 Explicit Update Rule for Multi-Neuron Independent Heads
Because we use **independent Sigmoid units** (enabling multi-label ambivalence like high Green + high Red), training does not require complex matrix backpropagation. It is formulated as **$K$ parallel perceptron updates** sharing the same bag-of-words input vector ($x_i \in \{0, 1\}$).

For each neuron $k$ (where $k \in \{\text{pos}, \text{neg}, \text{neu}\}$ for Tone, and $k \in \{\text{support}, \text{billing}, \text{sales}\}$ for Department):

1. **Forward Prediction:**
   $$z_k = b_k + \sum_{i} (w_{i,k} \cdot x_i)$$
   $$\hat{y}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

2. **Error Calculation:**
   $$\text{error}_k = y_k - \hat{y}_k$$
   *(where $y_k = 1.0$ if the sample belongs to class $k$, and $0.0$ otherwise).*

3. **Stochastic Gradient Descent (SGD) Update:**
   $$w_{i,k} \leftarrow w_{i,k} + (\alpha \cdot \text{error}_k \cdot x_i)$$
   $$b_k \leftarrow b_k + (\alpha \cdot \text{error}_k)$$
   *(with learning rate $\alpha = 0.15$).*

In Phase 3, this update executes across 6 parallel units per training sample. The loop is under 30 lines of readable vanilla JavaScript, perfectly auditable by non-programmers.

---

## 6. UI/UX & Layout Architecture

### 6.1 Top Navigation Bar
* Stage Switcher:
  * **[ 1. Black Text on White Space ]** (1 Neuron - Binary Teeter-Totter)
  * **[ 2. Red, Green, and Blue ]** (3 Neurons - Color Spectrum Sentiment)
  * **[ 3. Three Lanes in Two Dimensions ]** (Multi-Task - Tone & Department Routing)
* Subtitle & Context: 1-sentence plain-English mental model.

### 6.2 Main Split Workspace
* **Left Panel: Model Brain (Memory & Dial Matrix)**
  * **Phase 1 View:** 1-Neuron Weights table ($w_i$) and bias ($b$). Toggle between Raw Probability ($0 \to 1$) and Centered Polarity ($-1 \to +1$).
  * **Phase 2 View:** 3-Column Weights table showing Pos (🟢), Neg (🔴), and Neu (🔵) weights per word.
  * **Phase 3 View (UI Density Management):**
    * Sub-tabs: **[ 🎭 Tone Weights (3 Dials) ]** vs. **[ 🏢 Department Weights (3 Dials) ]**.
    * **Word Radar Inspector:** Clicking any word in the test sentence highlights its specific 6-dial profile (3 Tone weights, 3 Department weights).
* **Right Panel: Live Playground & Evaluation**
  * Sentence input with targeted test presets (including deliberate failure-mode tests like factual statements and mixed reviews).
  * **Phase 1 Gauge:** Dual display (0–100% Probability vs. -1 to +1 Polarity/Confidence).
  * **Phase 2 Gauge:** 3-channel RGB meter bar showing Green/Red/Blue channel intensities + combined color swatch.
  * **Phase 3 Gauge:** $3 \times 3$ Enterprise Decision Grid (Tone on Y-axis $\times$ Department on X-axis), plus an automated Action Banner (e.g., *"🔴 Negative + 💳 Billing $\implies$ Immediate Refund Escalation"*).
  * Live Mathematical Step-by-Step Breakdown card.
* **Training Control Center & Modal Dialog**
  * Shows Training Status (Untrained vs. Trained).
  * Modal with interactive step-by-step sample review, live multi-head loss plot, training speed throttle, and instant reset.

---

## 7. Verification & Acceptance Criteria
1. **Phase 1 Validation:**
   * Untrained state outputs $50.0\%$ (or $0.0$ polarity).
   * Clear visual toggle between Raw Probability ($0 \to 1$) and Centered Polarity ($-1 \to +1$).
   * Factual test sentence clearly demonstrates the "indifference vs. coin toss" limitation.
2. **Phase 2 Validation:**
   * Pure positive triggers strong Green channel ($>85\%$).
   * Pure negative triggers strong Red channel ($>85\%$).
   * Factual sentence triggers strong Blue/Neutral channel ($>80\%$).
   * Mixed sentence demonstrates dual Green + Red activation (Ambivalent/Mixed).
3. **Phase 3 Validation:**
   * Test sentence accurately activates both the correct Tone head and Department head.
   * Multi-head weights table clearly highlights words with specialized affinities (e.g. *"invoice"* to Billing, *"crash"* to Support, *"quote"* to Sales).
   * Density management (sub-tabs and word inspector) prevents cognitive overload.
4. **Performance & Cleanliness:**
   * Zero external dependencies or network calls; executes instantly in any modern browser.
   * Clean, inspectable source code with no obfuscated abstractions.
