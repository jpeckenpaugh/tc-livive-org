// ==========================================
// Fixed Dataset (10 Positive, 10 Negative)
// ==========================================
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

// Helper: Tokenize sentence into unique lowercase clean words
function tokenize(sentence) {
  return sentence
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(word => word.length > 0);
}

// Math Helper: Sigmoid squashing function
function sigmoid(z) {
  // clamp z to avoid numerical overflow
  if (z > 40) return 1.0;
  if (z < -40) return 0.0;
  return 1 / (1 + Math.exp(-z));
}

// ==========================================
// Application State
// ==========================================
class LearningLabState {
  constructor() {
    this.currentStage = 1; // 1, 2, or 3
    this.vocabulary = new Set();
    this.tallyMap = new Map(); // word -> { posCount: number, negCount: number }
    
    // Weights & Model parameters
    this.weights = new Map(); // word -> float
    this.bias = 0.0;
    this.learningRate = 0.15;
    
    // Training runtime status
    this.epochsTrained = 0;
    this.targetEpochs = 20;
    this.currentEpoch = 0;
    this.currentSampleIndex = 0;
    this.isTrainingRunning = false;
    this.lossHistory = [];
    
    this.buildVocabularyAndTallies();
    this.resetWeights();
  }

  buildVocabularyAndTallies() {
    this.vocabulary.clear();
    this.tallyMap.clear();

    TRAINING_DATA.forEach(sample => {
      const words = tokenize(sample.text);
      words.forEach(word => {
        this.vocabulary.add(word);
        if (!this.tallyMap.has(word)) {
          this.tallyMap.set(word, { posCount: 0, negCount: 0 });
        }
        const record = this.tallyMap.get(word);
        if (sample.label === 1) {
          record.posCount++;
        } else {
          record.negCount++;
        }
      });
    });
  }

  resetWeights() {
    this.weights.clear();
    this.vocabulary.forEach(w => {
      this.weights.set(w, 0.0);
    });
    this.bias = 0.0;
    this.epochsTrained = 0;
    this.currentEpoch = 0;
    this.currentSampleIndex = 0;
    this.isTrainingRunning = false;
    this.lossHistory = [0.5]; // initial coin toss error
  }

  isTrained() {
    return this.epochsTrained > 0;
  }

  // --- Inference Engine ---
  // Stage 1: Tally ratio / score
  predictStage1(sentence) {
    const tokens = tokenize(sentence);
    let totalPos = 0;
    let totalNeg = 0;
    const tokenDetails = [];

    tokens.forEach(tok => {
      const record = this.tallyMap.get(tok) || { posCount: 0, negCount: 0 };
      totalPos += record.posCount;
      totalNeg += record.negCount;
      tokenDetails.push({
        word: tok,
        pos: record.posCount,
        neg: record.negCount,
        inVocab: this.tallyMap.has(tok)
      });
    });

    const sum = totalPos + totalNeg;
    let probability = 0.5;
    if (sum > 0) {
      probability = totalPos / sum;
    }

    return {
      tokens,
      tokenDetails,
      totalPos,
      totalNeg,
      probability
    };
  }

  // Stage 2 & 3: Logistic Regression (z = b + sum(w_i), P = sigmoid(z))
  predictStage2(sentence) {
    const tokens = tokenize(sentence);
    // Unique tokens present in sentence
    const uniqueTokens = Array.from(new Set(tokens));
    let z = this.bias;
    const tokenDetails = [];

    uniqueTokens.forEach(tok => {
      const weight = this.weights.has(tok) ? this.weights.get(tok) : 0.0;
      z += weight;
      tokenDetails.push({
        word: tok,
        weight: weight,
        inVocab: this.weights.has(tok)
      });
    });

    const probability = sigmoid(z);
    return {
      tokens: uniqueTokens,
      tokenDetails,
      bias: this.bias,
      z,
      probability
    };
  }

  // Single step training on current sample
  stepTrainingSample() {
    if (this.currentSampleIndex >= TRAINING_DATA.length) {
      this.currentSampleIndex = 0;
      this.currentEpoch++;
    }

    const sample = TRAINING_DATA[this.currentSampleIndex];
    const words = tokenize(sample.text);
    const uniqueWords = Array.from(new Set(words));

    // 1. Predict
    let z = this.bias;
    uniqueWords.forEach(w => {
      z += (this.weights.get(w) || 0.0);
    });
    const prob = sigmoid(z);

    // 2. Error
    const error = sample.label - prob;

    // 3. Update weights & bias: delta = alpha * error
    const delta = this.learningRate * error;
    this.bias += delta * 0.5; // slow bias update

    const updatedWords = [];
    uniqueWords.forEach(w => {
      const oldW = this.weights.get(w) || 0.0;
      const newW = oldW + delta;
      this.weights.set(w, newW);
      updatedWords.push({
        word: w,
        oldWeight: oldW,
        newWeight: newW,
        delta: delta
      });
    });

    this.currentSampleIndex++;
    if (this.currentSampleIndex >= TRAINING_DATA.length) {
      this.epochsTrained = Math.max(this.epochsTrained, this.currentEpoch + 1);
    }

    return {
      sample,
      prob,
      error,
      delta,
      updatedWords,
      currentEpoch: this.currentEpoch,
      currentSampleIndex: this.currentSampleIndex
    };
  }
}

// Instantiate state
const labState = new LearningLabState();

// ==========================================
// UI Elements
// ==========================================
const UI = {
  // Navigation
  stageButtons: document.querySelectorAll('.stage-btn'),
  bannerBadge: document.getElementById('banner-badge'),
  bannerTitle: document.getElementById('banner-title'),
  bannerDesc: document.getElementById('banner-desc'),

  // Brain Views
  brainView1: document.getElementById('brain-view-stage-1'),
  brainView2: document.getElementById('brain-view-stage-2'),
  brainView3: document.getElementById('brain-view-stage-3'),
  
  // Stage 1 Table
  tallyTableBody: document.getElementById('tally-table-body'),

  // Stage 2 Table & Stats
  weightsTableBody: document.getElementById('weights-table-body'),
  statBias: document.getElementById('stat-bias'),
  statModelStatus: document.getElementById('stat-model-status'),
  statEpochs: document.getElementById('stat-epochs'),
  weightFilter: document.getElementById('weight-filter'),
  sortDescBtn: document.getElementById('sort-weight-desc'),
  sortAscBtn: document.getElementById('sort-weight-asc'),
  sortAlphaBtn: document.getElementById('sort-weight-alpha'),

  // Stage 3 Neuron SVG
  neuronSvg: document.getElementById('neuron-svg'),
  neuronCaption: document.getElementById('neuron-caption'),

  // Playground Inference
  testInput: document.getElementById('test-input'),
  btnRunInference: document.getElementById('btn-run-inference'),
  presetButtons: document.querySelectorAll('.btn-preset'),
  resultPercentLabel: document.getElementById('result-percent-label'),
  gaugeMarker: document.getElementById('gauge-marker'),
  predictionVerdict: document.getElementById('prediction-verdict'),
  mathBreakdown: document.getElementById('math-breakdown'),

  // Training Section
  trainingStatusPill: document.getElementById('training-status-pill'),
  metaWeightsState: document.getElementById('meta-weights-state'),
  btnOpenTraining: document.getElementById('btn-open-training'),
  btnResetWeights: document.getElementById('btn-reset-weights'),

  // Modal
  dialog: document.getElementById('training-dialog'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  modalEpochCount: document.getElementById('modal-epoch-count'),
  modalSampleCount: document.getElementById('modal-sample-count'),
  modalErrorVal: document.getElementById('modal-error-val'),
  modalLoopStatus: document.getElementById('modal-loop-status'),
  sampleTargetBadge: document.getElementById('sample-target-badge'),
  sampleText: document.getElementById('sample-text'),
  calcStepPred: document.getElementById('calc-step-pred'),
  calcStepError: document.getElementById('calc-step-error'),
  calcStepDelta: document.getElementById('calc-step-delta'),
  adjustedWordsList: document.getElementById('adjusted-words-list'),
  lossCanvas: document.getElementById('loss-canvas'),
  trainingSpeed: document.getElementById('training-speed'),
  speedLabel: document.getElementById('speed-label'),
  btnModalStep: document.getElementById('btn-modal-step'),
  btnModalRun: document.getElementById('btn-modal-run'),
  btnModalReset: document.getElementById('btn-modal-reset')
};

// ==========================================
// Banner & Stage View Switcher
// ==========================================
const STAGE_CONFIG = {
  1: {
    badge: "Stage 1: Crawl",
    title: "Counting Observations: Why Data Matters",
    desc: "Words are tallied into Positive vs. Negative buckets. Inference checks if a test sentence contains more positive tally marks or negative tally marks."
  },
  2: {
    badge: "Stage 2: Walk",
    title: "Dials & The Training Loop: Logistic Regression (SGD)",
    desc: "Every word has a numeric dial ('weight'). Untrained, every dial is 0.0 (50% coin toss). As we train over mistakes, words like 'great' spin positive and 'terrible' spin negative."
  },
  3: {
    badge: "Stage 3: Jog",
    title: "The Single-Neuron Perceptron: Neural Network Foundation",
    desc: "Demystifying AI jargon: An artificial neuron is identical to Stage 2. Inputs multiply by dial weights, aggregate at a summation nucleus, and activate into a prediction."
  }
};

function switchStage(stageNum) {
  labState.currentStage = parseInt(stageNum);

  // Update tabs
  UI.stageButtons.forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.stage) === labState.currentStage);
  });

  // Update banner
  const config = STAGE_CONFIG[labState.currentStage];
  UI.bannerBadge.textContent = config.badge;
  UI.bannerTitle.textContent = config.title;
  UI.bannerDesc.textContent = config.desc;

  // Toggle Brain views
  UI.brainView1.classList.toggle('active', labState.currentStage === 1);
  UI.brainView2.classList.toggle('active', labState.currentStage === 2);
  UI.brainView3.classList.toggle('active', labState.currentStage === 3);

  // Refresh current inference & brain representation
  renderBrainView();
  runInference();
}

// ==========================================
// Brain Panel Rendering
// ==========================================
let currentWeightSort = "desc"; // "desc", "asc", "alpha"

function renderStage1TallyTable(highlightTokens = []) {
  const sortedWords = Array.from(labState.vocabulary).sort();
  const highlightSet = new Set(highlightTokens.map(w => w.toLowerCase()));

  UI.tallyTableBody.innerHTML = sortedWords.map(word => {
    const data = labState.tallyMap.get(word);
    const isHit = highlightSet.has(word);
    let tendencyTag = `<span class="weight-tag neutral">Neutral (0/0)</span>`;
    if (data.posCount > data.negCount) {
      tendencyTag = `<span class="weight-tag pos">+${data.posCount - data.negCount} Pos</span>`;
    } else if (data.negCount > data.posCount) {
      tendencyTag = `<span class="weight-tag neg">-${data.negCount - data.posCount} Neg</span>`;
    }

    return `
      <tr class="${isHit ? 'highlight-hit' : ''}">
        <td><strong>${word}</strong></td>
        <td class="num-col" style="color: var(--pos-color); font-weight: 600;">${data.posCount}</td>
        <td class="num-col" style="color: var(--neg-color); font-weight: 600;">${data.negCount}</td>
        <td>${tendencyTag}</td>
      </tr>
    `;
  }).join('');
}

function renderStage2WeightsTable(highlightTokens = [], flashWords = new Map()) {
  UI.statBias.textContent = (labState.bias >= 0 ? "+" : "") + labState.bias.toFixed(3);
  UI.statEpochs.textContent = labState.epochsTrained;
  
  const isTrained = labState.isTrained();
  UI.statModelStatus.textContent = isTrained ? "Trained" : "Untrained (All 0.0)";
  UI.statModelStatus.className = `status-badge ${isTrained ? 'trained' : ''}`;
  UI.metaWeightsState.textContent = isTrained ? `Trained (${labState.epochsTrained} epochs)` : "All 0.0 (Untrained)";
  UI.trainingStatusPill.textContent = isTrained ? "Trained Model Active" : "Ready to Train";

  const filterText = UI.weightFilter.value.trim().toLowerCase();
  const highlightSet = new Set(highlightTokens.map(w => w.toLowerCase()));

  let wordList = Array.from(labState.vocabulary).map(word => {
    return {
      word,
      weight: labState.weights.get(word) || 0.0
    };
  });

  if (filterText) {
    wordList = wordList.filter(item => item.word.includes(filterText));
  }

  // Sorting
  if (currentWeightSort === "desc") {
    wordList.sort((a, b) => b.weight - a.weight);
  } else if (currentWeightSort === "asc") {
    wordList.sort((a, b) => a.weight - b.weight);
  } else {
    wordList.sort((a, b) => a.word.localeCompare(b.word));
  }

  UI.weightsTableBody.innerHTML = wordList.map(item => {
    const isHit = highlightSet.has(item.word);
    const flashClass = flashWords.has(item.word) 
      ? (flashWords.get(item.word) > 0 ? 'flash-pos' : 'flash-neg') 
      : '';

    let tag = `<span class="weight-tag neutral">Zero (0.00)</span>`;
    if (item.weight > 0.001) {
      tag = `<span class="weight-tag pos">Positive (+${item.weight.toFixed(2)})</span>`;
    } else if (item.weight < -0.001) {
      tag = `<span class="weight-tag neg">Negative (${item.weight.toFixed(2)})</span>`;
    }

    return `
      <tr class="${isHit ? 'highlight-hit' : ''} ${flashClass}">
        <td><strong>${item.word}</strong></td>
        <td class="num-col font-mono" style="font-weight: 700; color: ${item.weight > 0 ? 'var(--pos-color)' : item.weight < 0 ? 'var(--neg-color)' : 'var(--text-muted)'}">
          ${item.weight >= 0 ? '+' : ''}${item.weight.toFixed(3)}
        </td>
        <td>${tag}</td>
      </tr>
    `;
  }).join('');
}

// Stage 3: Render Interactive Single-Neuron SVG
function renderStage3Neuron(predictionData) {
  const tokens = predictionData ? predictionData.tokens : [];
  const tokenDetails = predictionData ? predictionData.tokenDetails : [];
  const bias = labState.bias;
  const z = predictionData ? predictionData.z : bias;
  const prob = predictionData ? predictionData.probability : sigmoid(bias);

  // SVG dimensions: 760 x 480
  const width = 760;
  const height = 480;
  const nucleusX = 390;
  const nucleusY = 240;
  const outputX = 670;
  const outputY = 240;

  // Render input slots (up to 6 inputs visually)
  const displayTokens = tokenDetails.slice(0, 6);
  const inputCount = Math.max(displayTokens.length, 1);
  const inputSpacing = Math.min(65, (height - 100) / inputCount);
  const startY = nucleusY - ((inputCount - 1) * inputSpacing) / 2;

  let inputsSvg = "";
  let synapLinesSvg = "";

  displayTokens.forEach((tok, idx) => {
    const inX = 90;
    const inY = startY + (idx * inputSpacing);
    const weightVal = tok.weight;
    const weightColor = weightVal > 0.05 ? "var(--pos-color)" : weightVal < -0.05 ? "var(--neg-color)" : "#94a3b8";
    const lineColor = weightVal > 0.05 ? "rgba(16, 185, 129, 0.7)" : weightVal < -0.05 ? "rgba(239, 68, 68, 0.7)" : "rgba(148, 163, 184, 0.3)";
    const strokeWidth = Math.max(1.5, Math.min(5, 1.5 + Math.abs(weightVal) * 1.5));

    // Connection path to nucleus
    synapLinesSvg += `
      <path d="M ${inX + 50} ${inY} C ${inX + 160} ${inY}, ${nucleusX - 90} ${nucleusY}, ${nucleusX - 60} ${nucleusY}" 
            fill="none" stroke="${lineColor}" stroke-width="${strokeWidth}" />
      <text x="${inX + 130}" y="${(inY + nucleusY) / 2 - 6}" fill="${weightColor}" font-size="11" font-family="monospace" font-weight="bold">
        w=${weightVal >= 0 ? '+' : ''}${weightVal.toFixed(2)}
      </text>
    `;

    // Input node pill
    inputsSvg += `
      <g transform="translate(${inX}, ${inY})">
        <rect x="-45" y="-16" width="95" height="32" rx="6" fill="#1e293b" stroke="#3b82f6" stroke-width="1.5" />
        <text x="2" y="4" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="600">
          "${tok.word}"
        </text>
      </g>
    `;
  });

  if (displayTokens.length === 0) {
    // Idle placeholder
    inputsSvg += `
      <g transform="translate(90, ${nucleusY})">
        <rect x="-60" y="-18" width="120" height="36" rx="6" fill="#1e293b" stroke="#475569" stroke-width="1" />
        <text x="0" y="5" text-anchor="middle" fill="#94a3b8" font-size="12">(Type sentence)</text>
      </g>
    `;
    synapLinesSvg += `
      <line x1="150" y1="${nucleusY}" x2="${nucleusX - 60}" y2="${nucleusY}" stroke="#475569" stroke-width="2" stroke-dasharray="4" />
    `;
  }

  // Nucleus: Summation + Sigmoid Gate
  const nucleusSvg = `
    <!-- Nucleus Circle -->
    <g transform="translate(${nucleusX}, ${nucleusY})">
      <circle r="65" fill="#1e293b" stroke="#38bdf8" stroke-width="3" filter="drop-shadow(0px 0px 8px rgba(56, 189, 248, 0.4))" />
      
      <!-- Divider line in cell -->
      <line x1="0" y1="-65" x2="0" y2="65" stroke="#334155" stroke-width="2" />
      
      <!-- Left side: Summation sigma -->
      <text x="-32" y="-12" text-anchor="middle" fill="#94a3b8" font-size="13" font-weight="bold">Σ w·x + b</text>
      <text x="-32" y="14" text-anchor="middle" fill="#38bdf8" font-size="15" font-family="monospace" font-weight="bold">
        z=${z >= 0 ? '+' : ''}${z.toFixed(2)}
      </text>
      <text x="-32" y="34" text-anchor="middle" fill="#64748b" font-size="10">
        b=${bias >= 0 ? '+' : ''}${bias.toFixed(2)}
      </text>

      <!-- Right side: Sigmoid Activation -->
      <path d="M 12 18 Q 28 18 32 0 T 52 -18" fill="none" stroke="#60a5fa" stroke-width="2.5" />
      <text x="32" y="-28" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="bold">Sigmoid σ</text>
      <text x="32" y="34" text-anchor="middle" fill="#e2e8f0" font-size="12" font-family="monospace">
        ${(prob * 100).toFixed(0)}%
      </text>
    </g>
  `;

  // Output Synapse to Meter
  const probColor = prob >= 0.55 ? "var(--pos-color)" : prob <= 0.45 ? "var(--neg-color)" : "var(--neutral-color)";
  const outLineColor = prob >= 0.55 ? "rgba(16, 185, 129, 0.8)" : prob <= 0.45 ? "rgba(239, 68, 68, 0.8)" : "rgba(245, 158, 11, 0.8)";
  
  const outputSvg = `
    <!-- Connection from Nucleus to Output -->
    <path d="M ${nucleusX + 65} ${nucleusY} L ${outputX - 45} ${outputY}" 
          fill="none" stroke="${outLineColor}" stroke-width="4" />
    <polygon points="${outputX - 45},${outputY - 5} ${outputX - 35},${outputY} ${outputX - 45},${outputY + 5}" fill="${outLineColor}" />

    <!-- Output Probability Display -->
    <g transform="translate(${outputX}, ${outputY})">
      <rect x="-40" y="-36" width="85" height="72" rx="10" fill="#0f172a" stroke="${probColor}" stroke-width="2.5" />
      <text x="2" y="-12" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="bold">OUTPUT</text>
      <text x="2" y="14" text-anchor="middle" fill="${probColor}" font-size="18" font-family="monospace" font-weight="800">
        ${(prob * 100).toFixed(1)}%
      </text>
      <text x="2" y="28" text-anchor="middle" fill="#cbd5e1" font-size="9" font-weight="600">
        ${prob >= 0.55 ? 'Positive' : prob <= 0.45 ? 'Negative' : 'Neutral'}
      </text>
    </g>
  `;

  UI.neuronSvg.innerHTML = `
    ${synapLinesSvg}
    ${inputsSvg}
    ${nucleusSvg}
    ${outputSvg}
  `;
}

function renderBrainView(highlightTokens = [], flashWords = new Map(), predictionData = null) {
  if (labState.currentStage === 1) {
    renderStage1TallyTable(highlightTokens);
  } else if (labState.currentStage === 2) {
    renderStage2WeightsTable(highlightTokens, flashWords);
  } else if (labState.currentStage === 3) {
    renderStage3Neuron(predictionData);
  }
}

// ==========================================
// Inference Engine Execution & UI Updates
// ==========================================
function runInference() {
  const sentence = UI.testInput.value.trim();
  if (!sentence) return;

  if (labState.currentStage === 1) {
    // Stage 1: Tally ratio
    const result = labState.predictStage1(sentence);
    const pct = (result.probability * 100).toFixed(1);
    UI.resultPercentLabel.textContent = `${pct}%`;
    UI.gaugeMarker.style.left = `${pct}%`;

    // Verdict tag
    if (result.probability > 0.55) {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag positive">Predicted: Positive (${pct}%)</span>`;
    } else if (result.probability < 0.45) {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag negative">Predicted: Negative (${pct}%)</span>`;
    } else {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag neutral">Neutral Coin Toss (50.0%)</span>`;
    }

    // Step-by-step arithmetic breakdown
    const breakdownLines = [];
    breakdownLines.push(`Tokens Found: [ ${result.tokens.map(t => `"${t}"`).join(', ')} ]`);
    breakdownLines.push(`Positive Word Hits: ${result.totalPos}`);
    breakdownLines.push(`Negative Word Hits: ${result.totalNeg}`);
    if (result.totalPos + result.totalNeg === 0) {
      breakdownLines.push(`Calculation: 0 observations found in training vocabulary -> Default neutral = 50.0%`);
    } else {
      breakdownLines.push(`Calculation: Positive Ratio = ${result.totalPos} / (${result.totalPos} + ${result.totalNeg}) = ${result.probability.toFixed(3)} (${pct}%)`);
    }
    UI.mathBreakdown.innerHTML = breakdownLines.join('<br>');

    renderStage1TallyTable(result.tokens);

  } else {
    // Stage 2 & Stage 3: Logistic Regression / Single Neuron
    const result = labState.predictStage2(sentence);
    const pct = (result.probability * 100).toFixed(1);
    UI.resultPercentLabel.textContent = `${pct}%`;
    UI.gaugeMarker.style.left = `${pct}%`;

    // Verdict tag
    if (result.probability > 0.55) {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag positive">Predicted: Positive (${pct}%)</span>`;
    } else if (result.probability < 0.45) {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag negative">Predicted: Negative (${pct}%)</span>`;
    } else {
      UI.predictionVerdict.innerHTML = `<span class="verdict-tag neutral">Neutral Coin Toss (50.0%)</span>`;
    }

    // Step-by-step arithmetic breakdown
    const breakdownLines = [];
    const sumTerms = result.tokenDetails.map(t => {
      const sign = t.weight >= 0 ? "+" : "";
      return `${sign}${t.weight.toFixed(3)} ("${t.word}")`;
    });

    const biasStr = (result.bias >= 0 ? "+" : "") + result.bias.toFixed(3);
    const formulaStr = sumTerms.length > 0 
      ? `z = bias (${biasStr}) + ${sumTerms.join(" + ")} = <span class="math-highlight">${result.z.toFixed(3)}</span>`
      : `z = bias (${biasStr}) + 0.000 = <span class="math-highlight">${result.z.toFixed(3)}</span>`;

    breakdownLines.push(formulaStr);
    breakdownLines.push(`P = Sigmoid(z) = 1 / (1 + e^(${-result.z.toFixed(3)})) = <span class="math-highlight">${result.probability.toFixed(4)} (${pct}%)</span>`);
    
    if (!labState.isTrained()) {
      breakdownLines.push(`<span style="color: var(--neutral-color)">Note: All weights are currently 0.0 (Untrained state). Prediction is guaranteed 50.0%.</span>`);
    }

    UI.mathBreakdown.innerHTML = breakdownLines.join('<br>');

    if (labState.currentStage === 2) {
      renderStage2WeightsTable(result.tokens);
    } else {
      renderStage3Neuron(result);
    }
  }
}

// ==========================================
// Loss Chart (Canvas)
// ==========================================
function renderLossChart() {
  const canvas = UI.lossCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = 20; y < h; y += 25) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();

  const data = labState.lossHistory;
  if (data.length === 0) return;

  // Plot line
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  const stepX = (w - 40) / Math.max(data.length - 1, 1);
  data.forEach((val, i) => {
    const x = 20 + i * stepX;
    // val ranges 0.0 to 0.5 (or higher)
    const normalizedY = Math.max(0, Math.min(1, val / 0.6));
    const y = h - 20 - (normalizedY * (h - 40));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Highlight points
  data.forEach((val, i) => {
    const x = 20 + i * stepX;
    const normalizedY = Math.max(0, Math.min(1, val / 0.6));
    const y = h - 20 - (normalizedY * (h - 40));
    ctx.fillStyle = i === data.length - 1 ? "#10b981" : "#38bdf8";
    ctx.beginPath();
    ctx.arc(x, y, i === data.length - 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

// Compute average loss over entire dataset
function calculateDatasetMeanError() {
  let totalAbsError = 0;
  TRAINING_DATA.forEach(sample => {
    const pred = labState.predictStage2(sample.text).probability;
    totalAbsError += Math.abs(sample.label - pred);
  });
  return totalAbsError / TRAINING_DATA.length;
}

// ==========================================
// Modal Training Loop Logic
// ==========================================
function updateModalInspectors(stepInfo) {
  UI.modalEpochCount.textContent = `${stepInfo.currentEpoch + 1} / ${labState.targetEpochs}`;
  UI.modalSampleCount.textContent = `${stepInfo.currentSampleIndex} / ${TRAINING_DATA.length}`;
  
  const meanErr = calculateDatasetMeanError();
  UI.modalErrorVal.textContent = meanErr.toFixed(3);

  const sample = stepInfo.sample;
  UI.sampleText.textContent = `"${sample.text}"`;
  UI.sampleTargetBadge.textContent = sample.label === 1 ? "Target: 1 (Positive)" : "Target: 0 (Negative)";
  UI.sampleTargetBadge.className = `sample-target-badge ${sample.label === 1 ? '' : 'neg'}`;

  UI.calcStepPred.textContent = `P = Sigmoid(z) = ${stepInfo.prob.toFixed(3)}`;
  UI.calcStepError.textContent = `error = ${sample.label} - ${stepInfo.prob.toFixed(3)} = ${stepInfo.error >= 0 ? '+' : ''}${stepInfo.error.toFixed(3)}`;
  UI.calcStepDelta.textContent = `Δw = 0.15 × (${stepInfo.error >= 0 ? '+' : ''}${stepInfo.error.toFixed(3)}) = ${stepInfo.delta >= 0 ? '+' : ''}${stepInfo.delta.toFixed(3)}`;

  UI.adjustedWordsList.innerHTML = stepInfo.updatedWords.map(uw => {
    const color = uw.delta > 0 ? 'var(--pos-color)' : 'var(--neg-color)';
    return `<span class="adj-pill" style="color: ${color}">"${uw.word}": ${uw.newWeight.toFixed(2)}</span>`;
  }).join(' ');

  // Flash updated words in weights table
  const flashMap = new Map();
  stepInfo.updatedWords.forEach(uw => flashMap.set(uw.word, uw.delta));
  renderStage2WeightsTable([], flashMap);
}

function executeSingleStep() {
  const stepInfo = labState.stepTrainingSample();
  updateModalInspectors(stepInfo);

  // If completed an epoch, record loss history
  if (stepInfo.currentSampleIndex === TRAINING_DATA.length) {
    const meanErr = calculateDatasetMeanError();
    labState.lossHistory.push(meanErr);
    renderLossChart();
  }

  runInference();
}

async function runAllTrainingEpochs() {
  if (labState.isTrainingRunning) return;
  labState.isTrainingRunning = true;
  UI.btnModalRun.disabled = true;
  UI.btnModalStep.disabled = true;
  UI.modalLoopStatus.textContent = "Training...";
  UI.modalLoopStatus.className = "mstat-value status-badge";

  const totalSteps = labState.targetEpochs * TRAINING_DATA.length;
  let executedSteps = labState.currentEpoch * TRAINING_DATA.length + labState.currentSampleIndex;

  while (executedSteps < totalSteps && labState.isTrainingRunning) {
    const stepInfo = labState.stepTrainingSample();
    executedSteps++;

    // Periodic UI update according to slider
    const delay = parseInt(UI.trainingSpeed.value);
    
    // Only update inspectors every step if delay > 50ms, else batch UI
    if (delay > 50 || executedSteps % 5 === 0 || executedSteps === totalSteps) {
      updateModalInspectors(stepInfo);
      if (stepInfo.currentSampleIndex === TRAINING_DATA.length) {
        const meanErr = calculateDatasetMeanError();
        labState.lossHistory.push(meanErr);
        renderLossChart();
      }
      runInference();
      await new Promise(r => setTimeout(r, delay));
    }
  }

  labState.isTrainingRunning = false;
  UI.btnModalRun.disabled = false;
  UI.btnModalStep.disabled = false;
  UI.modalLoopStatus.textContent = "Finished!";
  UI.modalLoopStatus.className = "mstat-value status-badge trained";

  const finalMeanErr = calculateDatasetMeanError();
  labState.lossHistory.push(finalMeanErr);
  renderLossChart();
  runInference();
}

// Reset weights and UI
function resetWeightsAndLab() {
  labState.resetWeights();
  UI.modalLoopStatus.textContent = "Idle (Untrained)";
  UI.modalLoopStatus.className = "mstat-value status-badge";
  UI.modalEpochCount.textContent = "0 / 20";
  UI.modalSampleCount.textContent = "0 / 20";
  UI.modalErrorVal.textContent = "0.500";
  UI.adjustedWordsList.innerHTML = "";
  renderLossChart();
  renderBrainView();
  runInference();
}

// ==========================================
// Event Listeners & Initialization
// ==========================================
function setupEventListeners() {
  // Stage Selector Tabs
  UI.stageButtons.forEach(btn => {
    btn.addEventListener('click', () => switchStage(btn.dataset.stage));
  });

  // Inference Input & Evaluation
  UI.btnRunInference.addEventListener('click', runInference);
  UI.testInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runInference();
  });
  UI.testInput.addEventListener('input', () => {
    if (labState.currentStage === 3) {
      // Dynamic response in neuron diagram as user types
      runInference();
    }
  });

  // Presets
  UI.presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      UI.testInput.value = btn.dataset.text;
      runInference();
    });
  });

  // Sorting and Filtering Weights Table
  UI.weightFilter.addEventListener('input', () => renderStage2WeightsTable());
  UI.sortDescBtn.addEventListener('click', () => {
    currentWeightSort = "desc";
    renderStage2WeightsTable();
  });
  UI.sortAscBtn.addEventListener('click', () => {
    currentWeightSort = "asc";
    renderStage2WeightsTable();
  });
  UI.sortAlphaBtn.addEventListener('click', () => {
    currentWeightSort = "alpha";
    renderStage2WeightsTable();
  });

  // Modal Dialog controls
  UI.btnOpenTraining.addEventListener('click', () => {
    UI.dialog.showModal();
    renderLossChart();
  });
  UI.btnCloseModal.addEventListener('click', () => {
    labState.isTrainingRunning = false;
    UI.dialog.close();
  });
  UI.dialog.addEventListener('cancel', () => {
    labState.isTrainingRunning = false;
  });

  // Modal Action Buttons
  UI.btnModalStep.addEventListener('click', executeSingleStep);
  UI.btnModalRun.addEventListener('click', runAllTrainingEpochs);
  UI.btnModalReset.addEventListener('click', resetWeightsAndLab);
  UI.btnResetWeights.addEventListener('click', resetWeightsAndLab);

  // Speed Slider
  UI.trainingSpeed.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    if (val < 40) UI.speedLabel.textContent = "Instant";
    else if (val < 120) UI.speedLabel.textContent = "Fast";
    else if (val < 250) UI.speedLabel.textContent = "Normal";
    else UI.speedLabel.textContent = "Slow";
  });
}

// Initial Boot
function init() {
  setupEventListeners();
  switchStage(1);
  resetWeightsAndLab();
}

window.addEventListener('DOMContentLoaded', init);
