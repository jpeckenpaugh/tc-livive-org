#!/usr/bin/env python3
"""
Showcase recorder and gallery capture script for Training vs. Inference (tc-livive-org).
Standardized 30-second 720p demo video + 6 high-resolution gallery screenshots.
"""

import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = ROOT / "demos" / "latest"
GALLERY_DIR = DEMOS_DIR / "gallery"
PORT = 4192

# Ensure directories exist
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logging

def start_server():
    server = socketserver.TCPServer(("127.0.0.1", PORT), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server

def inject_cursor(page):
    js_code = """
    (() => {
        if (window.__cursorInjected) return;
        window.__cursorInjected = true;
        const cursor = document.createElement('div');
        cursor.id = 'playwright-demo-cursor';
        cursor.style.position = 'fixed';
        cursor.style.top = '0';
        cursor.style.left = '0';
        cursor.style.width = '20px';
        cursor.style.height = '20px';
        cursor.style.borderRadius = '50%';
        cursor.style.backgroundColor = 'rgba(56, 189, 248, 0.7)';
        cursor.style.border = '2px solid #ffffff';
        cursor.style.boxShadow = '0 0 10px rgba(56, 189, 248, 0.8)';
        cursor.style.pointerEvents = 'none';
        cursor.style.zIndex = '9999999';
        cursor.style.transform = 'translate(-50%, -50%)';
        cursor.style.transition = 'width 0.15s ease, height 0.15s ease, background-color 0.15s ease';
        document.body.appendChild(cursor);

        window.__setCursorPos = (x, y) => {
            cursor.style.left = x + 'px';
            cursor.style.top = y + 'px';
        };

        window.__cursorClickAnim = () => {
            cursor.style.width = '30px';
            cursor.style.height = '30px';
            cursor.style.backgroundColor = 'rgba(239, 68, 68, 0.9)';
            setTimeout(() => {
                cursor.style.width = '20px';
                cursor.style.height = '20px';
                cursor.style.backgroundColor = 'rgba(56, 189, 248, 0.7)';
            }, 200);
        };
    })();
    """
    page.evaluate(js_code)

def move_mouse_smooth(page, from_x, from_y, to_x, to_y, steps=12, interval=0.015):
    for i in range(1, steps + 1):
        curr_x = from_x + (to_x - from_x) * (i / steps)
        curr_y = from_y + (to_y - from_y) * (i / steps)
        page.mouse.move(curr_x, curr_y)
        page.evaluate(f"window.__setCursorPos && window.__setCursorPos({curr_x}, {curr_y})")
        time.sleep(interval)

def click_element_smooth(page, selector, current_pos, wait_after=0.2):
    elem = page.locator(selector).first
    elem.scroll_into_view_if_needed()
    box = elem.bounding_box()
    if box:
        target_x = box['x'] + box['width'] / 2
        target_y = box['y'] + box['height'] / 2
        move_mouse_smooth(page, current_pos[0], current_pos[1], target_x, target_y)
        page.evaluate("window.__cursorClickAnim && window.__cursorClickAnim()")
        time.sleep(0.08)
        elem.click()
        if wait_after > 0:
            time.sleep(wait_after)
        return (target_x, target_y)
    return current_pos

def main():
    print(f"[*] Starting local server on http://127.0.0.1:{PORT} ...")
    server = start_server()
    time.sleep(0.5)

    video_tmp_dir = ROOT / "demos" / "tmp_video"
    video_tmp_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Launching Playwright Chromium (1280x720, 30-second target) ...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=1,
            record_video_dir=str(video_tmp_dir),
            record_video_size={"width": 1280, "height": 720}
        )
        page = context.new_page()

        start_time = time.time()
        pos = (640, 360)

        # Navigate
        page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        inject_cursor(page)
        page.evaluate(f"window.__setCursorPos && window.__setCursorPos({pos[0]}, {pos[1]})")

        # ----------------------------------------------------
        # Scene 1: Orientation Primer Slide Deck (00s - 04.5s)
        # ----------------------------------------------------
        print("[1/6] Capturing Slide Deck Orientation...")
        # Wait for dialog to auto-open (or click open button if needed)
        try:
            page.wait_for_selector("#walkthrough-dialog[open]", timeout=1500)
        except Exception:
            pos = click_element_smooth(page, "#btn-open-walkthrough", pos)
            page.wait_for_selector("#walkthrough-dialog[open]", timeout=2000)

        inject_cursor(page)
        time.sleep(0.3)

        # Advance through slides to Slide 4 (Architecture: Distillation into Memory)
        for s in range(3):
            pos = click_element_smooth(page, "#btn-wt-next", pos, wait_after=0.35)

        time.sleep(0.5)
        # Still 1: Architecture modal
        page.screenshot(path=str(GALLERY_DIR / "01-orientation-architecture.png"))
        print("  -> Saved 01-orientation-architecture.png")

        # Close modal
        pos = click_element_smooth(page, "#btn-close-walkthrough", pos, wait_after=0.5)

        # ----------------------------------------------------
        # Scene 2: View 1 - The Blank Slate Baseline (04.5s - 08.5s)
        # ----------------------------------------------------
        print("[2/6] Demonstrating View 1: Blank Slate Baseline...")
        # Hover RAM status
        pos = click_element_smooth(page, "#ram-status-pill", pos, wait_after=0.4)

        # Click test sentence
        pos = click_element_smooth(page, ".test-card-btn", pos, wait_after=0.6)

        # Still 2: Blank slate 50/50 baseline
        page.screenshot(path=str(GALLERY_DIR / "02-blank-slate-baseline.png"))
        print("  -> Saved 02-blank-slate-baseline.png")
        time.sleep(1.0)

        # ----------------------------------------------------
        # Scene 3: View 2 - 2-Sample Model Training & Blind Spot (08.5s - 15.5s)
        # ----------------------------------------------------
        print("[3/6] Demonstrating View 2: Training & Blind Spot Discovery...")
        pos = click_element_smooth(page, "#crawl-btn-step-2", pos, wait_after=0.6)

        # Click Train
        pos = click_element_smooth(page, "#btn-train-disk", pos, wait_after=0.6)

        # Click Load into RAM
        pos = click_element_smooth(page, "#btn-load-ram", pos, wait_after=0.8)

        # Still 3: Train and load pipeline
        page.screenshot(path=str(GALLERY_DIR / "03-train-and-load-pipeline.png"))
        print("  -> Saved 03-train-and-load-pipeline.png")

        # Click Test 1 (Positive match: great)
        pos = click_element_smooth(page, "[data-testid='s2-1']", pos, wait_after=0.8)

        # Click Test 4 (Blind spot: broken and damaged)
        pos = click_element_smooth(page, "[data-testid='s2-4']", pos, wait_after=0.8)

        # Still 4: Out of vocabulary blind spot
        page.screenshot(path=str(GALLERY_DIR / "04-out-of-vocabulary-blind-spot.png"))
        print("  -> Saved 04-out-of-vocabulary-blind-spot.png")
        time.sleep(0.8)

        # ----------------------------------------------------
        # Scene 4: View 3 - Scaling Data & Token Expansion (15.5s - 22.5s)
        # ----------------------------------------------------
        print("[4/6] Demonstrating View 3: Scaling Token Coverage...")
        pos = click_element_smooth(page, "#crawl-btn-step-3", pos, wait_after=0.6)

        # Train on 10 samples
        pos = click_element_smooth(page, "#btn-train-disk", pos, wait_after=0.6)

        # Load 38 tokens into RAM
        pos = click_element_smooth(page, "#btn-load-ram", pos, wait_after=0.8)

        # Still 5: Expanded RAM table
        page.screenshot(path=str(GALLERY_DIR / "05-scaling-memory-coverage.png"))
        print("  -> Saved 05-scaling-memory-coverage.png")

        # Click Test 4 again to show blind spot resolved!
        pos = click_element_smooth(page, "[data-testid='s3-4']", pos, wait_after=1.2)

        # ----------------------------------------------------
        # Scene 5: View 4 - Interactive Playground & Math Trace (22.5s - 30.0s)
        # ----------------------------------------------------
        print("[5/6] Demonstrating View 4: Live Custom Evaluator...")
        pos = click_element_smooth(page, "#crawl-btn-step-4", pos, wait_after=0.6)

        # Focus input and type
        pos = click_element_smooth(page, "#crawl-user-input", pos, wait_after=0.2)
        page.fill("#crawl-user-input", "")
        time.sleep(0.2)
        test_text = "The service was fast and great communication."
        for char in test_text:
            page.type("#crawl-user-input", char, delay=35)
        time.sleep(0.3)

        pos = click_element_smooth(page, "#btn-crawl-eval-user", pos, wait_after=0.8)

        # Still 6: Final math inspector
        page.screenshot(path=str(GALLERY_DIR / "06-live-math-inspector.png"))
        print("  -> Saved 06-live-math-inspector.png")

        # Hover over math trace until exactly 30s elapsed
        pos = click_element_smooth(page, "#crawl-math-trace", pos, wait_after=0.2)
        elapsed = time.time() - start_time
        remaining = 30.0 - elapsed
        if remaining > 0:
            print(f"[*] Polishing final frame hold ({remaining:.1f}s) to hit 30.0s runtime...")
            time.sleep(remaining)

        # Finish recording
        print("[*] Closing browser session to flush video stream...")
        video_path = page.video.path()
        context.close()
        browser.close()

    server.shutdown()

    # Move and transcode video
    print(f"[*] Raw video saved at: {video_path}")
    final_webm = DEMOS_DIR / "demo.webm"

    # Use ffmpeg if available to optimize and enforce exact 30s duration
    ffmpeg_bin = Path(os.path.expanduser("~/Library/Caches/ms-playwright/ffmpeg-1013/ffmpeg-mac"))
    if ffmpeg_bin.exists():
        print(f"[*] Transcoding with {ffmpeg_bin} to {final_webm} ...")
        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-i", str(video_path),
            "-t", "30.0",
            "-c:v", "libvpx_vp9",
            "-b:v", "750k",
            "-crf", "32",
            "-an",
            str(final_webm)
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"[*] Moving raw video to {final_webm} ...")
        os.replace(video_path, final_webm)

    # Generate webp thumbnail
    thumb_path = DEMOS_DIR / "thumb.webp"
    print(f"[*] Generating optimized WebP thumbnail at {thumb_path} ...")
    with Image.open(GALLERY_DIR / "06-live-math-inspector.png") as img:
        img_rgb = img.convert("RGB")
        img_rgb.save(thumb_path, "WEBP", quality=85)

    print("[SUCCESS] Showcase assets created successfully:")
    print(f"  - Video: {final_webm} ({os.path.getsize(final_webm)} bytes)")
    print(f"  - Thumbnail: {thumb_path} ({os.path.getsize(thumb_path)} bytes)")
    for g in sorted(GALLERY_DIR.glob("*.png")):
        print(f"  - Gallery: {g.name} ({os.path.getsize(g)} bytes)")

if __name__ == "__main__":
    main()
