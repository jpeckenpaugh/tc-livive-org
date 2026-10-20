// =======================================================================
// TROT ONNX RUNNER
// A dependency-free ONNX reader + executor for the four-node TROT graph:
//   input -> Gemm(W1,b1,transB=1) -> Tanh -> Gemm(W2,b2,transB=1) -> Sigmoid
//
// It reads the actual bytes of models/trot.onnx (embedded in onnx-data.js
// so the page also works from file://). No ONNX Runtime, no CDN, no iframe.
// =======================================================================
(function () {
  "use strict";

  // ---- protobuf primitives ------------------------------------------------

  function readVarint(bytes, pos) {
    let value = 0;
    let shift = 0;
    let byte;
    do {
      byte = bytes[pos++];
      value |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    return [value >>> 0, pos];
  }

  function* fields(bytes, start = 0, end = bytes.length) {
    let pos = start;
    while (pos < end) {
      let tag;
      [tag, pos] = readVarint(bytes, pos);
      const field = tag >>> 3;
      const wire = tag & 7;
      if (wire === 0) {
        let value;
        [value, pos] = readVarint(bytes, pos);
        yield { field, wire, value };
      } else if (wire === 2) {
        let len;
        [len, pos] = readVarint(bytes, pos);
        yield { field, wire, value: bytes.subarray(pos, pos + len) };
        pos += len;
      } else if (wire === 5) {
        yield { field, wire, value: bytes.subarray(pos, pos + 4) };
        pos += 4;
      } else if (wire === 1) {
        yield { field, wire, value: bytes.subarray(pos, pos + 8) };
        pos += 8;
      } else {
        throw new Error(`Unsupported protobuf wire type ${wire}`);
      }
    }
  }

  const utf8 = (bytes) => new TextDecoder().decode(bytes);

  function parseTensor(bytes) {
    const tensor = { dims: [], name: "", dataType: 0, raw: null };
    for (const f of fields(bytes)) {
      if (f.field === 1) tensor.dims.push(f.value);
      else if (f.field === 2) tensor.dataType = f.value;
      else if (f.field === 8) tensor.name = utf8(f.value);
      else if (f.field === 9) tensor.raw = f.value;
    }
    if (!tensor.raw) throw new Error(`Tensor ${tensor.name} has no raw_data`);
    if (tensor.dataType !== 1) throw new Error(`Tensor ${tensor.name} is not float32`);
    const copy = tensor.raw.slice();
    tensor.data = new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
    return tensor;
  }

  function parseNode(bytes) {
    const node = { inputs: [], outputs: [], op: "", attrs: {} };
    for (const f of fields(bytes)) {
      if (f.field === 1) node.inputs.push(utf8(f.value));
      else if (f.field === 2) node.outputs.push(utf8(f.value));
      else if (f.field === 4) node.op = utf8(f.value);
      else if (f.field === 5) {
        let name = "";
        const ints = {};
        for (const a of fields(f.value)) {
          if (a.field === 1) name = utf8(a.value);
          else if (a.field === 3) ints[name] = a.value;
        }
        if (name) node.attrs[name] = ints[name];
      }
    }
    return node;
  }

  function parseValueInfo(bytes) {
    let name = "";
    for (const f of fields(bytes)) {
      if (f.field === 1) name = utf8(f.value);
    }
    return name;
  }

  function parseGraph(bytes) {
    const graph = { name: "", nodes: [], initializers: {}, inputs: [], outputs: [] };
    for (const f of fields(bytes)) {
      if (f.field === 1) graph.nodes.push(parseNode(f.value));
      else if (f.field === 2) graph.name = utf8(f.value);
      else if (f.field === 5) {
        const tensor = parseTensor(f.value);
        graph.initializers[tensor.name] = tensor;
      } else if (f.field === 11) graph.inputs.push(parseValueInfo(f.value));
      else if (f.field === 12) graph.outputs.push(parseValueInfo(f.value));
    }
    return graph;
  }

  function parseModel(bytes) {
    const model = { irVersion: 0, producer: "", opset: 0, graph: null };
    for (const f of fields(bytes)) {
      if (f.field === 1) model.irVersion = f.value;
      else if (f.field === 2) model.producer = utf8(f.value);
      else if (f.field === 7) model.graph = parseGraph(f.value);
      else if (f.field === 8) {
        for (const o of fields(f.value)) {
          if (o.field === 2) model.opset = o.value;
        }
      }
    }
    return model;
  }

  function base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // ---- inference ---------------------------------------------------------

  function tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .split(/\s+/)
      .filter(Boolean);
  }

  function vectorize(sentence, vocab) {
    const tokens = tokenize(sentence);
    const x = new Float32Array(vocab.length);
    const active = tokens.map((token) => {
      const index = vocab.indexOf(token);
      if (index >= 0) x[index] = 1;
      return { token, index, known: index >= 0 };
    });
    return { tokens, active, x };
  }

  function runVector(model, x) {
    const init = model.graph.initializers;
    const W1 = init.W1;
    const b1 = init.b1;
    const W2 = init.W2;
    const b2 = init.b2;
    if (!W1 || !b1 || !W2 || !b2) throw new Error("Missing TROT initializers");

    const hidden = W1.dims[0];
    const vocab = W1.dims[1];
    if (x.length !== vocab) throw new Error(`Expected ${vocab} inputs, got ${x.length}`);

    // Gemm: z1 = x · W1ᵀ + b1  (transB = 1)
    const z1 = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) {
      let acc = b1.data[j];
      const row = j * vocab;
      for (let i = 0; i < vocab; i++) acc += W1.data[row + i] * x[i];
      z1[j] = acc;
    }

    // Tanh
    const h = new Float32Array(hidden);
    for (let j = 0; j < hidden; j++) h[j] = Math.tanh(z1[j]);

    // Gemm: z2 = h · W2ᵀ + b2  (transB = 1)
    let z2 = b2.data[0];
    for (let j = 0; j < hidden; j++) z2 += W2.data[j] * h[j];

    // Sigmoid
    const p = 1 / (1 + Math.exp(-z2));

    return { hidden, vocab, z1, h, z2, p };
  }

  function runSentence(model, vocab, sentence) {
    const vec = vectorize(sentence, vocab);
    return { ...vec, ...runVector(model, vec.x) };
  }

  // ---- UI ----------------------------------------------------------------

  const q = (id) => document.getElementById(id);
  const fmt = (n, d = 4) => Number(n).toFixed(d);

  function renderGraph(model, bytes) {
    const init = model.graph.initializers;
    const rows = [];
    rows.push(
      `<div class="onnx-graph-row"><span class="onnx-op">input</span><span>[1, ${init.W1.dims[1]}]</span></div>`
    );
    for (const node of model.graph.nodes) {
      const attrs = Object.entries(node.attrs)
        .map(([k, v]) => `${k}=${v}`)
        .join(" ");
      const shapes = node.inputs
        .map((name) => (init[name] ? `${name}[${init[name].dims.join("×")}]` : name))
        .join(" · ");
      rows.push(
        `<div class="onnx-graph-row"><span class="onnx-op">${node.op}</span><span>${shapes}${
          attrs ? ` · ${attrs}` : ""
        }</span></div>`
      );
    }
    rows.push(
      `<div class="onnx-graph-row"><span class="onnx-op">output</span><span>${model.graph.outputs[0]} · [batch, 1]</span></div>`
    );
    q("onnx-graph").innerHTML = rows.join("");
    q("onnx-status").textContent = `${bytes.length.toLocaleString()} bytes · IR ${
      model.irVersion
    } · opset ${model.opset}`;
  }

  function renderTokens(result) {
    q("onnx-tokens").innerHTML = result.active
      .map(
        (a) =>
          `<span class="chip ${a.known ? "chip-hit" : "chip-oov"}" title="${
            a.known ? `index ${a.index} in the 50-token vocabulary` : "not in the vocabulary"
          }">${a.token}${a.known ? " ✓" : " · OOV"}</span>`
      )
      .join(" ");
  }

  function renderSteps(result) {
    const vec = (label, arr) =>
      `<div class="onnx-step"><div class="onnx-step-label">${label}</div><div class="onnx-step-values">${Array.from(
        arr,
        (v) => `<span>${fmt(v)}</span>`
      ).join("")}</div></div>`;

    q("onnx-steps").innerHTML = [
      vec("z1 = W1x + b1  (8)", result.z1),
      vec("h = tanh(z1)  (8)", result.h),
      `<div class="onnx-step"><div class="onnx-step-label">z2 = W2h + b2</div><div class="onnx-step-values"><span>${fmt(
        result.z2,
        5
      )}</span></div></div>`,
      `<div class="onnx-step"><div class="onnx-step-label">p = σ(z2)</div><div class="onnx-step-values"><span>${fmt(
        result.p,
        6
      )}</span></div></div>`,
    ].join("");
  }

  function renderResult(result) {
    const pct = (result.p * 100).toFixed(1);
    q("onnx-prob").textContent = `${pct}%`;
    q("onnx-prob").className = `onnx-prob ${result.p >= 0.5 ? "pos" : "neg"}`;
    renderTokens(result);
    renderSteps(result);
  }

  function init() {
    const data = window.TROT_ONNX_DATA;
    if (!data || !q("onnx-runner")) return;

    let bytes;
    let model;
    try {
      bytes = base64ToBytes(data.bytesBase64);
      model = parseModel(bytes);
    } catch (error) {
      q("onnx-status").textContent = "model error";
      q("onnx-verify-note").textContent = `Could not parse the ONNX file: ${error.message}`;
      return;
    }

    const vocab = data.vocab;
    const input = q("onnx-input");

    const run = () => {
      const text = input.value.trim();
      if (!text) return;
      renderResult(runSentence(model, vocab, text));
    };

    q("btn-onnx-run").addEventListener("click", run);
    input.addEventListener("input", run);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") run();
    });

    document.querySelectorAll("[data-onnx-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.onnxPreset;
        run();
      });
    });

    q("btn-onnx-download").addEventListener("click", () => {
      const blob = new Blob([bytes], { type: "application/octet-stream" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "trot.onnx";
      link.click();
      URL.revokeObjectURL(url);
    });

    q("btn-onnx-verify").addEventListener("click", () => {
      const ref = data.reference;
      let maxDiff = 0;
      let worst = -1;
      ref.inputs.forEach((vector, index) => {
        const result = runVector(model, vector);
        const diff = Math.abs(result.p - ref.probs[index]);
        if (diff > maxDiff) {
          maxDiff = diff;
          worst = index;
        }
      });
      q("onnx-verify-note").innerHTML =
        `✓ ${ref.inputs.length}/${ref.inputs.length} predictions match the Rust core. ` +
        `Max |JS − Rust| = <strong>${maxDiff.toExponential(2)}</strong>` +
        (worst >= 0 ? ` (row ${worst})` : "") +
        `. The .onnx executor really is running the exported graph.`;
    });

    renderGraph(model, bytes);
    renderResult(runSentence(model, vocab, input.value));
  }

  window.TrotOnnx = {
    parseModel,
    base64ToBytes,
    tokenize,
    vectorize,
    runVector,
    runSentence,
    init,
  };
})();
