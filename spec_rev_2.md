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
    * 🟢 **Green Neuron (Positive):** $z_{\text{pos}} = b_{\text{pos}} + \sum w_{i,\text{pos}} x_i \to P_{\text{pos}}$
    * 🔴 **Red Neuron (Negative):** $z_{\text{neg}} = b_{\text{neg}} + \sum w_{i,\text{neg}} x_i \to P_{\text{neg}}$
    * 🔵 **Blue Neuron (Neutral):** $z_{\text{neu}} = b_{\text{neu}} + \sum w_{i,\text{neu}} x_i \to P_{\text{neu}}$
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
  * Weights table expands into a clean, 2-dimensional inspectable matrix:
    * Words like *"refund"*, *"charge"*, *"invoice"* activate the **Billing** lane.
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

## 4. Dataset Specification (Rev 2 Expanded Training Set)

To train both the 3-sentiment head and the 3-department head, the dataset contains ~30 balanced, realistic business feedback snippets:

### Dataset Schema
Each sample contains:
* `text`: Clean snippet.
* `sentiment`: `pos` | `neg` | `neu`
* `department`: `support` | `billing` | `sales`

### Training Samples (Representative Breakdown)
1. **Support + Positive:** *"support agent quickly resolved the bug"*
2. **Support + Negative:** *"terrible app keeps crashing on login"*
3. **Support + Neutral:** *"how do I reset my password on mobile"*
4. **Billing + Positive:** *"prompt refund received thank you very much"*
5. **Billing + Negative:** *"unauthorized double charge on my credit card"*
6. **Billing + Neutral:** *"where can I download my annual tax receipt"*
7. **Sales + Positive:** *"excited to upgrade our contract to enterprise tier"*
8. **Sales + Negative:** *"pricing is completely unreasonable for small teams"*
9. **Sales + Neutral:** *"requesting a formal price quote for ten seats"*
*(Full dataset contains 30 samples to ensure distinct keyword distributions).*

---

## 5. UI/UX & Layout Architecture

### 5.1 Top Navigation Bar
* Stage Switcher:
  * **[ 1. Black Text on White Space ]** (1 Neuron - Binary Teeter-Totter)
  * **[ 2. Red, Green, and Blue ]** (3 Neurons - Color Spectrum Sentiment)
  * **[ 3. Three Lanes in Two Dimensions ]** (Multi-Task - Tone & Department Routing)
* Subtitle & Context: 1-sentence plain-English mental model.

### 5.2 Main Split Workspace
* **Left Panel: Model Brain (Memory & Dial Matrix)**
  * Phase 1: 1-Neuron Weights table ($w_i$) and bias ($b$). Polarity display toggle.
  * Phase 2: 3-Column Weights table showing Pos, Neg, and Neu weights for each word.
  * Phase 3: Interactive Matrix View showing weights across Tone (Pos/Neg/Neu) and Routing (Support/Billing/Sales).
* **Right Panel: Live Playground & Evaluation**
  * Sentence input with targeted test presets (including deliberate failure-mode tests like factual statements and mixed reviews).
  * Phase 1 Gauge: Dual display (0–100% Probability vs. -1 to +1 Polarity/Confidence).
  * Phase 2 Gauge: 3-channel RGB meter bar showing Green/Red/Blue channel intensities.
  * Phase 3 Gauge: 2D Grid with routed department and tone recommendation banner.
  * Live Mathematical Step-by-Step Breakdown card.
* **Training Control Center & Modal Dialog**
  * Shows Training Status (Untrained vs. Trained).
  * Modal with interactive step-by-step sample review, live loss plot, training speed throttle, and instant reset.

---

## 6. Verification & Acceptance Criteria
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
4. **Performance & Cleanliness:**
   * Zero external dependencies or network calls; executes instantly in any modern browser.
   * Clean, inspectable source code with no obfuscated abstractions.
