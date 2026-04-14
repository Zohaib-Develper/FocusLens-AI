(() => {
  "use strict";

  const API_URL = "http://localhost:8000/predict";
  const CAPTURE_INTERVAL = 3000;
  const JPEG_QUALITY = 0.7;
  const IMG_SIZE = 224;

  let captureTimer = null;
  let isRunning = false;

  // reusable canvas for frame extraction
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");

  function captureFrame(video) {
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return null;
    try {
      ctx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
      return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    } catch {
      return null; // cross-origin blocked
    }
  }

  async function predict(base64Image) {
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });
      if (!res.ok) throw new Error(res.status);
      return await res.json();
    } catch (err) {
      console.warn("[FocusLens] prediction failed:", err.message);
      return null;
    }
  }

  // walks up the DOM to find the participant tile wrapping a video
  function findTileContainer(video) {
    let el = video.parentElement;
    for (let i = 0; i < 5 && el; i++) {
      if (el.offsetWidth >= 100 && el.offsetHeight >= 100) return el;
      el = el.parentElement;
    }
    return el;
  }

  function discoverParticipants() {
    const results = [];
    document.querySelectorAll("video").forEach((video) => {
      if (video.videoWidth < 50 || video.offsetWidth < 50) return;
      const container = findTileContainer(video);
      if (container) results.push({ video, container });
    });
    return results;
  }

  // create or update the overlay on a participant tile
  function updateOverlay(container, result) {
    let overlay = container.querySelector(".fl-overlay");

    if (!overlay) {
      overlay = document.createElement("div");
      overlay.className = "fl-overlay";
      overlay.innerHTML = `
        <div class="fl-badge">
          <span class="fl-icon"></span>
          <span class="fl-status"></span>
        </div>
        <div class="fl-confidence">
          <div class="fl-bar"><div class="fl-bar-fill"></div></div>
          <span class="fl-bar-text"></span>
        </div>
        <div class="fl-probs">
          <div class="fl-prob-row">
            <span>Attentive</span>
            <span class="fl-att-val"></span>
          </div>
          <div class="fl-prob-row">
            <span>Distracted</span>
            <span class="fl-dis-val"></span>
          </div>
        </div>
      `;

      // make sure container can hold the overlay
      if (getComputedStyle(container).position === "static") {
        container.style.position = "relative";
      }
      container.appendChild(overlay);
    }

    const attentive = result.prediction === "Attentive";
    const conf = (result.confidence * 100).toFixed(1);

    overlay.classList.toggle("fl-attentive", attentive);
    overlay.classList.toggle("fl-distracted", !attentive);

    overlay.querySelector(".fl-icon").textContent = attentive ? "👁️" : "😴";
    overlay.querySelector(".fl-status").textContent = result.prediction;
    overlay.querySelector(".fl-bar-text").textContent = `${conf}%`;
    overlay.querySelector(".fl-bar-fill").style.width = `${conf}%`;

    const att = (result.probabilities.Attentive * 100).toFixed(1);
    const dis = (result.probabilities.Distracted * 100).toFixed(1);
    overlay.querySelector(".fl-att-val").textContent = `${att}%`;
    overlay.querySelector(".fl-dis-val").textContent = `${dis}%`;

    // glow effect on the tile
    container.classList.remove("fl-glow-green", "fl-glow-red");
    container.classList.add(attentive ? "fl-glow-green" : "fl-glow-red");
  }

  async function runCaptureCycle() {
    const participants = discoverParticipants();
    if (participants.length === 0) return;

    const tasks = participants.map(async ({ video, container }) => {
      const frame = captureFrame(video);
      if (!frame) return;

      const result = await predict(frame);
      if (!result || result.error) return;

      updateOverlay(container, result);
    });

    await Promise.allSettled(tasks);
  }

  function start() {
    if (isRunning) return;
    isRunning = true;
    console.log("[FocusLens AI] started monitoring");

    setTimeout(runCaptureCycle, 2000);
    captureTimer = setInterval(runCaptureCycle, CAPTURE_INTERVAL);
  }

  function stop() {
    if (!isRunning) return;
    isRunning = false;
    clearInterval(captureTimer);
    captureTimer = null;

    // clean up overlays
    document.querySelectorAll(".fl-overlay").forEach((el) => el.remove());
    document.querySelectorAll(".fl-glow-green, .fl-glow-red").forEach((el) => {
      el.classList.remove("fl-glow-green", "fl-glow-red");
    });

    console.log("[FocusLens AI] stopped");
  }

  // watch for video elements appearing/disappearing (join/leave call)
  const observer = new MutationObserver(() => {
    const hasVideos = document.querySelectorAll("video").length > 0;
    if (hasVideos && !isRunning) start();
    else if (!hasVideos && isRunning) stop();
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // in case videos already exist
  if (document.querySelectorAll("video").length > 0) start();

  window.addEventListener("beforeunload", () => {
    stop();
    observer.disconnect();
  });

  console.log("[FocusLens AI] extension loaded, waiting for video streams...");
})();
