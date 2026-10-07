// =======================================================================
// TRAINING VS. INFERENCE LEARNING LAB
// Core Engine implementing Module 1 (Reframed 3-View Hardware Architecture)
// =======================================================================

// --- Tokenization Helper ---
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter(Boolean);
}

// --- Module 1 Datasets & Test Scenarios ---
const CRAWL_DATASETS = {
  step1: {
    diskFiles: [],
    tests: [
      { id: "s1-a", text: "The product arrived fast and works great." },
      { id: "s1-b", text: "The product was broken and customer service was awful." }
    ]
  },
  step2: {
    diskFiles: [
      { text: "The service was great and fast.", label: "positive" },
      { text: "The service was terrible and slow.", label: "negative" }
    ],
    tests: [
      { id: "s2-1", text: "They have great communication." },
      { id: "s2-2", text: "Shipping was incredibly slow." },
      { id: "s2-3", text: "I really loved the experience." },
      { id: "s2-4", text: "The package arrived broken and damaged." }
    ]
  },
  step3: {
    diskFiles: [
      // 5 Positive
      { text: "The service was great and fast.", label: "positive" },
      { text: "I really loved the wonderful experience.", label: "positive" },
      { text: "Friendly support team resolved my issue immediately.", label: "positive" },
      { text: "Customer service was excelente and solved my issue fast.", label: "positive" },
      { text: "The build quality is truly magnifique.", label: "positive" },
      // 5 Negative
      { text: "The service was terrible and slow.", label: "negative" },
      { text: "The package was broken and arrived damaged.", label: "negative" },
      { text: "Awful quality broke immediately on day one.", label: "negative" },
      { text: "Completely useless product and rude customer support.", label: "negative" },
      { text: "Disappointing experience with constant crashes.", label: "negative" }
    ],
    tests: [
      { id: "s3-1", text: "They have great communication." },
      { id: "s3-2", text: "Shipping was incredibly slow." },
      { id: "s3-3", text: "I really loved the experience." },
      { id: "s3-4", text: "The package arrived broken and damaged." }
    ]
  },
  step4: {
    diskFiles: [
      // 4 Focused Samples (2 Positive / 2 Negative)
      { text: "The service was great and fast.", label: "positive" },
      { text: "Helpful team solved my issue with fantastic communication.", label: "positive" },
      { text: "The service was terrible and slow.", label: "negative" },
      { text: "Awful support broke my account with frustrating delay.", label: "negative" }
    ],
    tests: [
      { id: "s4-1", text: "Great communication and fast resolution." },
      { id: "s4-2", text: "Terrible delay and slow response." },
      { id: "s4-3", text: "The package was delivered on Tuesday." }
    ]
  }
};

// =======================================================================
// State Controller for Module 1 (Hardware & Memory)
// =======================================================================
class CrawlHardwareLab {
  constructor() {
    this.currentStep = 1; // 1, 2, or 3
    this.modelCompiledOnDisk = false;
    this.modelLoadedInRAM = false;
    this.compiledModelArtifact = null; // { [token]: { pos: N, neg: N } }
    this.activeRamLookup = {}; // Active lookup table in RAM
    this.activeTestSentence = "";
    this.activeTestId = "";
  }

  setStep(step) {
    this.currentStep = parseInt(step);
    this.modelCompiledOnDisk = false;
    this.modelLoadedInRAM = false;
    this.compiledModelArtifact = null;
    this.activeRamLookup = {};

    const dataset = CRAWL_DATASETS[`step${this.currentStep}`];
    this.activeTestSentence = dataset.tests[0].text;
    this.activeTestId = dataset.tests[0].id;

    // View 4 starts with model pre-trained and loaded in RAM for immediate interactive exploration
    if (this.currentStep === 4) {
      this.trainModel();
      this.loadModelToRAM();
      if (DOM.crawlUserInput && DOM.crawlUserInput.value) {
        this.activeTestSentence = DOM.crawlUserInput.value;
        this.activeTestId = "custom-user";
      }
    }
  }

  // Action 1: Train Model (reads raw data on disk, tokenizes, counts tallies)
  trainModel() {
    const dataset = CRAWL_DATASETS[`step${this.currentStep}`];
    if (dataset.diskFiles.length === 0) return null;

    const artifact = {};
    dataset.diskFiles.forEach(sample => {
      const tokens = tokenize(sample.text);
      tokens.forEach(tok => {
        if (!artifact[tok]) {
          artifact[tok] = { pos: 0, neg: 0 };
        }
        if (sample.label === "positive") artifact[tok].pos += 1;
        if (sample.label === "negative") artifact[tok].neg += 1;
      });
    });

    this.compiledModelArtifact = artifact;
    this.modelCompiledOnDisk = true;
    return artifact;
  }

  // Action 2: Load Model into RAM (simple I/O transfer into active memory)
  loadModelToRAM() {
    if (!this.compiledModelArtifact) return false;
    // Deep copy into RAM lookup table
    this.activeRamLookup = JSON.parse(JSON.stringify(this.compiledModelArtifact));
    this.modelLoadedInRAM = true;
    return true;
  }

  // Inference Execution: evaluates sentence against activeRamLookup
  evaluateInference(sentence) {
    const tokens = tokenize(sentence);
    let totalPos = 0;
    let totalNeg = 0;
    const tokenDetails = [];

    tokens.forEach(tok => {
      const match = this.activeRamLookup[tok];
      if (match) {
        totalPos += match.pos;
        totalNeg += match.neg;
        tokenDetails.push({
          token: tok,
          pos: match.pos,
          neg: match.neg,
          inRAM: true
        });
      } else {
        tokenDetails.push({
          token: tok,
          pos: 0,
          neg: 0,
          inRAM: false
        });
      }
    });

    const totalClues = totalPos + totalNeg;
    let probability = 0.5; // strictly 50% if 0 clues
    if (totalClues > 0) {
      probability = totalPos / totalClues;
    }

    return {
      sentence,
      tokens,
      tokenDetails,
      totalPos,
      totalNeg,
      totalClues,
      probability
    };
  }
}

// Global Lab Instance
const crawlLab = new CrawlHardwareLab();

// =======================================================================
// UI Bindings & DOM References
// =======================================================================
const DOM = {
  // Top Module Tabs
  moduleButtons: document.querySelectorAll('.stage-btn'),
  moduleContainers: {
    crawl: document.getElementById('module-crawl-container'),
    walk: document.getElementById('module-walk-container'),
    jog: document.getElementById('module-jog-container'),
    run: document.getElementById('module-run-container')
  },

  // Crawl Step Sub-navigation
  crawlStepButtons: document.querySelectorAll('.crawl-step-btn'),
  crawlBannerBadge: document.getElementById('crawl-banner-badge'),
  crawlBannerTitle: document.getElementById('crawl-banner-title'),
  crawlBannerDesc: document.getElementById('crawl-banner-desc'),

  // Hardware Left Panel
  ramStatusPill: document.getElementById('ram-status-pill'),
  diskStatusBadge: document.getElementById('disk-status-badge'),
  diskDescText: document.getElementById('disk-desc-text'),
  diskSamplesList: document.getElementById('disk-samples-list'),
  btnTrainDisk: document.getElementById('btn-train-disk'),
  btnLoadRAM: document.getElementById('btn-load-ram'),
  trainStepIndicator: document.getElementById('train-step-indicator'),
  ramCountBadge: document.getElementById('ram-count-badge'),
  ramTableBody: document.getElementById('ram-table-body'),

  // Inference Right Panel
  crawlCustomInputCard: document.getElementById('crawl-custom-input-card'),
  crawlUserInput: document.getElementById('crawl-user-input'),
  btnCrawlEvalUser: document.getElementById('btn-crawl-eval-user'),
  crawlTestCards: document.getElementById('crawl-test-cards'),
  crawlVerdictTag: document.getElementById('crawl-verdict-tag'),
  crawlTokenDisplay: document.getElementById('crawl-token-display'),
  crawlScoreLabel: document.getElementById('crawl-score-label'),
  crawlGaugeMarker: document.getElementById('crawl-gauge-marker'),
  crawlMathTrace: document.getElementById('crawl-math-trace')
};

// =======================================================================
// Step Metadata & Explanations
// =======================================================================
const STEP_META = {
  1: {
    badge: "View 1: The Blank Slate",
    title: "Empty RAM: Baseline Guessing",
    desc: "Computers do not inherently 'know English' or human emotions. Without a compiled model loaded in RAM, the processor has zero reference clues and must resort to an uncalibrated 50/50 coin toss."
  },
  2: {
    badge: "View 2: The 2-Sample Model",
    title: "Train ➔ Load ➔ Generalization & Blind Spots",
    desc: "Training compiles raw text into an artifact; loading puts it into RAM. The model generalizes cleanly on shared words ('great', 'slow'), but is totally blind to unobserved words."
  },
  3: {
    badge: "View 3: Scaling the Data",
    title: "Scaling Data ➔ Token Coverage & Loanword Reality",
    desc: "Training on 10 samples expands RAM vocabulary (~40 tokens) to eliminate blind spots. Notice how loanwords ('excelente', 'magnifique') are stored as plain string tokens."
  },
  4: {
    badge: "View 4: Try It for Yourself",
    title: "4-Sample Model: Interactive Playground",
    desc: "Type any sentence you like! Observe in real time which words hit the active RAM lookup table, which words become blind spots (gray pills), and how the CPU calculates the score."
  }
};

// =======================================================================
// UI Rendering Functions
// =======================================================================

function renderCrawlView() {
  const step = crawlLab.currentStep;
  const meta = STEP_META[step];
  const dataset = CRAWL_DATASETS[`step${step}`];

  // Update step buttons active state
  DOM.crawlStepButtons.forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.step) === step);
  });

  // Update banners
  DOM.crawlBannerBadge.textContent = meta.badge;
  DOM.crawlBannerTitle.textContent = meta.title;
  DOM.crawlBannerDesc.textContent = meta.desc;

  // --- Render Disk Storage Section ---
  if (step === 1) {
    DOM.diskStatusBadge.textContent = "0 raw files";
    DOM.diskStatusBadge.classList.remove('active');
    DOM.diskDescText.textContent = "No training data on disk yet. The computer has no source material to learn from.";
    DOM.diskSamplesList.innerHTML = `<div style="color: var(--text-dim); font-size: 0.8rem; text-align: center; padding: 1rem;">(Disk storage contains 0 training datasets)</div>`;
    DOM.btnTrainDisk.disabled = true;
    DOM.btnLoadRAM.disabled = true;
    DOM.trainStepIndicator.textContent = "";
  } else {
    DOM.diskStatusBadge.textContent = `${dataset.diskFiles.length} raw samples on disk`;
    DOM.diskStatusBadge.classList.add('active');
    DOM.diskDescText.textContent = `Raw unparsed sentences stored on disk. Click 'Train' to compile them into a dictionary artifact.`;
    DOM.diskSamplesList.innerHTML = dataset.diskFiles.map(s => `
      <div class="disk-sample-item">
        <span class="disk-sample-tag ${s.label === 'positive' ? 'pos' : 'neg'}">${s.label === 'positive' ? '+ POS' : '- NEG'}</span>
        <span style="color: var(--text-main); font-style: italic;">"${s.text}"</span>
      </div>
    `).join('');

    DOM.btnTrainDisk.disabled = false;
    DOM.btnLoadRAM.disabled = !crawlLab.modelCompiledOnDisk;

    if (!crawlLab.modelCompiledOnDisk) {
      DOM.trainStepIndicator.textContent = "Ready to train";
    } else if (crawlLab.modelCompiledOnDisk && !crawlLab.modelLoadedInRAM) {
      DOM.trainStepIndicator.textContent = "✓ Model compiled to disk! Now load to RAM.";
    } else {
      DOM.trainStepIndicator.textContent = "✓ Model active in RAM!";
    }
  }

  // --- Render RAM Table Section ---
  const activeTokens = Object.keys(crawlLab.activeRamLookup);
  if (activeTokens.length === 0) {
    DOM.ramStatusPill.textContent = "RAM: 0 Tokens (Empty)";
    DOM.ramCountBadge.textContent = "0 Active Lookups";
    DOM.ramTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="4" style="text-align: center; color: var(--text-dim); padding: 2rem;">
          RAM is empty. Load a model to inspect active memory tokens.
        </td>
      </tr>
    `;
  } else {
    DOM.ramStatusPill.textContent = `RAM: ${activeTokens.length} Tokens Loaded`;
    DOM.ramCountBadge.textContent = `${activeTokens.length} Active Lookups`;
    
    DOM.ramTableBody.innerHTML = activeTokens.sort().map(tok => {
      const data = crawlLab.activeRamLookup[tok];
      let leanTag = `<span class="weight-tag neutral">Tied (1/1)</span>`;
      if (data.pos > data.neg) {
        leanTag = `<span class="weight-tag pos">+${data.pos - data.neg} Pos</span>`;
      } else if (data.neg > data.pos) {
        leanTag = `<span class="weight-tag neg">-${data.neg - data.pos} Neg</span>`;
      }

      return `
        <tr>
          <td><strong>${tok}</strong></td>
          <td class="num-col" style="color: var(--pos-color); font-weight: 600;">${data.pos}</td>
          <td class="num-col" style="color: var(--neg-color); font-weight: 600;">${data.neg}</td>
          <td>${leanTag}</td>
        </tr>
      `;
    }).join('');
  }

  // --- Render Custom Input or Presets ---
  if (step === 4) {
    DOM.crawlCustomInputCard.style.display = "flex";
  } else {
    DOM.crawlCustomInputCard.style.display = "none";
  }

  // --- Render Test Preset Cards ---
  DOM.crawlTestCards.innerHTML = dataset.tests.map(t => {
    const isSelected = t.id === crawlLab.activeTestId;
    return `
      <div class="test-card-btn ${isSelected ? 'active' : ''}" data-testid="${t.id}" data-text="${t.text}">
        <span class="test-card-text">"${t.text}"</span>
        <span class="test-card-badge ${isSelected ? 'pos' : 'neutral'}">
          ${isSelected ? 'Evaluating ▶' : 'Test'}
        </span>
      </div>
    `;
  }).join('');

  // Re-attach test button listeners
  document.querySelectorAll('.test-card-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      crawlLab.activeTestId = btn.dataset.testid;
      crawlLab.activeTestSentence = btn.dataset.text;
      if (step === 4) {
        DOM.crawlUserInput.value = btn.dataset.text;
      }
      renderCrawlView();
    });
  });

  // Execute and render active test inference
  renderInferenceResult();
}

function renderInferenceResult() {
  const result = crawlLab.evaluateInference(crawlLab.activeTestSentence);
  const pct = (result.probability * 100).toFixed(1);

  // Update Score & Gauge
  DOM.crawlScoreLabel.textContent = `${pct}%`;
  DOM.crawlGaugeMarker.style.left = `${pct}%`;

  // Update Verdict Tag
  if (result.totalClues === 0) {
    DOM.crawlVerdictTag.className = "verdict-tag neutral";
    DOM.crawlVerdictTag.textContent = "50.0% (Undecided / 0 Clues in RAM)";
  } else if (result.probability > 0.55) {
    DOM.crawlVerdictTag.className = "verdict-tag positive";
    DOM.crawlVerdictTag.textContent = `Predicted Positive (${pct}%)`;
  } else if (result.probability < 0.45) {
    DOM.crawlVerdictTag.className = "verdict-tag negative";
    DOM.crawlVerdictTag.textContent = `Predicted Negative (${pct}%)`;
  } else {
    DOM.crawlVerdictTag.className = "verdict-tag neutral";
    DOM.crawlVerdictTag.textContent = `Tied Evidence (${pct}%)`;
  }

  // Render Token Highlighting Pills (Green/Red for hits, Gray for OOV)
  DOM.crawlTokenDisplay.innerHTML = result.tokenDetails.map(item => {
    if (!item.inRAM) {
      return `<span class="tok-pill oov" title="Out of Vocabulary: Not in RAM">"${item.token}" (0)</span>`;
    }
    if (item.pos > item.neg) {
      return `<span class="tok-pill pos" title="Positive in RAM">"${item.token}" (+${item.pos})</span>`;
    }
    if (item.neg > item.pos) {
      return `<span class="tok-pill neg" title="Negative in RAM">"${item.token}" (-${item.neg})</span>`;
    }
    return `<span class="tok-pill neutral" title="Tied evidence">"${item.token}" (=)</span>`;
  }).join(' ');

  // Mathematical Trace
  const recognizedTokens = result.tokenDetails.filter(t => t.inRAM);
  const traceLines = [];

  if (recognizedTokens.length === 0) {
    traceLines.push(`Recognized Tokens in RAM: [ none ]`);
    traceLines.push(`Positive Evidence (P): 0 | Negative Evidence (N): 0`);
    traceLines.push(`Score Calculation: 0 clues found in active memory ➔ Default 50.0% (Coin Toss)`);
  } else {
    traceLines.push(`Recognized Tokens in RAM: [ ${recognizedTokens.map(t => `"${t.token}"`).join(', ')} ]`);
    traceLines.push(`Positive Evidence (P): ${result.totalPos} | Negative Evidence (N): ${result.totalNeg}`);
    traceLines.push(`Score Calculation: P / (P + N) = ${result.totalPos} / (${result.totalPos} + ${result.totalNeg}) = ${result.probability.toFixed(3)} ➔ ${pct}%`);
  }

  DOM.crawlMathTrace.innerHTML = traceLines.join('<br>');
}

// =======================================================================
// Event Listeners & Bootstrapping
// =======================================================================
function setupCrawlEventListeners() {
  // Module Switching
  DOM.moduleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetModule = btn.dataset.module;
      DOM.moduleButtons.forEach(b => b.classList.toggle('active', b === btn));
      Object.keys(DOM.moduleContainers).forEach(m => {
        DOM.moduleContainers[m].classList.toggle('active', m === targetModule);
      });
    });
  });

  // Step Sub-navigation
  DOM.crawlStepButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      crawlLab.setStep(btn.dataset.step);
      renderCrawlView();
    });
  });

  // Train Button
  DOM.btnTrainDisk.addEventListener('click', () => {
    crawlLab.trainModel();
    renderCrawlView();
  });

  // Load into RAM Button
  DOM.btnLoadRAM.addEventListener('click', () => {
    crawlLab.loadModelToRAM();
    renderCrawlView();
  });

  // Custom User Input Evaluation in View 4
  function evaluateCustomInput() {
    const text = DOM.crawlUserInput.value.trim();
    if (!text) return;
    crawlLab.activeTestSentence = text;
    crawlLab.activeTestId = "custom-user";
    renderCrawlView();
  }

  DOM.btnCrawlEvalUser.addEventListener('click', evaluateCustomInput);
  DOM.crawlUserInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') evaluateCustomInput();
  });
  DOM.crawlUserInput.addEventListener('input', () => {
    // Real-time evaluation as the user types
    const text = DOM.crawlUserInput.value.trim();
    if (text) {
      crawlLab.activeTestSentence = text;
      crawlLab.activeTestId = "custom-user";
      renderInferenceResult();
    }
  });
}

function init() {
  setupCrawlEventListeners();
  crawlLab.setStep(1);
  renderCrawlView();
}

window.addEventListener('DOMContentLoaded', init);
