import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';

const root = process.env.APP_ROOT || '/app';
const output = '/artifacts';
const port = 4173;
const showcaseMode = process.env.TC_SHOWCASE === '1';
const errors = [];
let browser;
let context;
let server;

const result = {
  status: 'failed',
  mode: showcaseMode ? 'showcase' : 'fast-smoke',
  startedAt: new Date().toISOString(),
  browser: null,
  steps: [],
  errors,
};

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const record = (step) => result.steps.push({ step, status: 'passed' });
const snap = (page, name) => page.screenshot({ path: path.join(output, name), fullPage: true });

try {
  await fs.mkdir(output, { recursive: true });
  server = http.createServer(async (req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      const file = path.resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
      assert(file.startsWith(path.resolve(root) + path.sep), 'Invalid static path');
      const data = await fs.readFile(file);
      const ext = path.extname(file);
      res.writeHead(200, {
        'Content-Type':
          ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml' })[ext] ||
          'application/octet-stream',
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  await new Promise((resolve) => server.listen(port, '0.0.0.0', resolve));

  browser = await chromium.launch({ headless: true });
  result.browser = {
    playwright: '1.55.0',
    chromium: browser.version(),
    viewport: { width: 1280, height: 720 },
  };
  context = await browser.newContext({
    viewport: result.browser.viewport,
    ...(showcaseMode ? { recordVideo: { dir: output, size: result.browser.viewport } } : {}),
  });

  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });

  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
  record('page loaded');

  // Verify Orientation Dialog
  await page.waitForSelector('#walkthrough-dialog[open]', { timeout: 3000 });
  await snap(page, '01-orientation-architecture.png');
  record('orientation dialog rendered');

  await page.locator('#btn-close-walkthrough').click();
  await page.waitForTimeout(400);

  // View 1
  await page.locator('.test-card-btn').first().click();
  await snap(page, '02-blank-slate-baseline.png');
  record('view 1 blank slate baseline verified');

  // View 2
  await page.locator('#crawl-btn-step-2').click();
  await page.locator('#btn-train-disk').click();
  await page.locator('#btn-load-ram').click();
  await snap(page, '03-train-and-load-pipeline.png');
  record('view 2 train and load verified');

  await page.locator("[data-testid='s2-4']").click();
  await snap(page, '04-out-of-vocabulary-blind-spot.png');
  record('view 2 out of vocabulary blind spot verified');

  // View 3
  await page.locator('#crawl-btn-step-3').click();
  await page.locator('#btn-train-disk').click();
  await page.locator('#btn-load-ram').click();
  await page.locator("[data-testid='s3-4']").click();
  await snap(page, '05-scaling-memory-coverage.png');
  record('view 3 scaling memory coverage verified');

  // View 4
  await page.locator('#crawl-btn-step-4').click();
  await page.locator('#crawl-user-input').fill('The service was fast and great communication.');
  await page.locator('#btn-crawl-eval-user').click();
  await snap(page, '06-live-math-inspector.png');
  record('view 4 custom user evaluation verified');

  if (showcaseMode) {
    await page.waitForTimeout(3000);
  }

  result.status = errors.length === 0 ? 'passed' : 'failed';
} catch (error) {
  errors.push(`fatal: ${error.message}`);
} finally {
  if (context) await context.close();
  if (browser) await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
  await fs.writeFile(path.join(output, 'results.json'), JSON.stringify(result, null, 2));
}

if (result.status !== 'passed') {
  process.exitCode = 1;
}
