// multiple_video_comparison.js
//
// Generalization of refnerf's `video_comparison.js`
// (https://dorverbin.github.io/refnerf/) from 2 panes to N panes.
// The source video is N method renders hstacked horizontally
// (so its width = N * paneW). One canvas displays paneW wide; N-1 draggable
// seams partition the canvas into N regions, each region showing the
// corresponding pane of the video. Drag any seam to wipe between adjacent
// methods. Labels are read from the video's `data-labels` attribute as a
// pipe-separated string ("Ours|Baseline A|Baseline B|...").

if (Number.prototype.clamp === undefined) {
    Number.prototype.clamp = function (min, max) {
        return Math.min(Math.max(this, min), max);
    };
}

function playMultiVids(videoId) {
    const canvas = document.getElementById(videoId + "Merge");
    const vid = document.getElementById(videoId);
    if (!canvas || !vid) {
        return;
    }
    const nPanes = parseInt(vid.dataset.panes || "2", 10);
    const labels = (vid.dataset.labels || "").split("|");
    const ctx = canvas.getContext("2d");

    if (vid.readyState <= 3) {
        setTimeout(function () { playMultiVids(videoId); }, 100);
        return;
    }
    vid.play();

    const paneW = vid.videoWidth / nPanes;
    const H = vid.videoHeight;

    // Initial seams. Default = equally spaced inside [0, paneW]; can be
    // overridden via `data-init-seams="f1,f2,..."` on the <video> tag, where
    // each f_i is a fraction in (0, 1) giving seam_i's position as a fraction
    // of paneW. Useful for biasing one pane to take more canvas width by
    // default (e.g. pin "Ours" to the central 50%).
    const seams = [];
    const initAttr = (vid.dataset.initSeams || "").trim();
    let initFracs = null;
    if (initAttr) {
        const parsed = initAttr.split(",").map(s => parseFloat(s));
        if (parsed.length === nPanes - 1 && parsed.every(v => Number.isFinite(v) && v > 0 && v < 1)) {
            initFracs = parsed;
        }
    }
    for (let i = 1; i < nPanes; i++) {
        const frac = initFracs ? initFracs[i - 1] : (i / nPanes);
        seams.push(frac * paneW);
    }

    let draggingIdx = -1;
    const DRAG_RADIUS_PX = 28;

    function clientToCanvasX(clientX) {
        const bcr = canvas.getBoundingClientRect();
        const x = ((clientX - bcr.left) / bcr.width) * paneW;
        return x.clamp(0, paneW);
    }

    function nearestSeam(canvasX) {
        const bcr = canvas.getBoundingClientRect();
        const pxPerCanvas = bcr.width / paneW;
        let best = -1, bestDist = Infinity;
        for (let i = 0; i < seams.length; i++) {
            const d = Math.abs(seams[i] - canvasX) * pxPerCanvas;
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return bestDist < DRAG_RADIUS_PX ? best : -1;
    }

    function onDown(e) {
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        draggingIdx = nearestSeam(clientToCanvasX(cx));
    }
    function onMove(e) {
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const canvasX = clientToCanvasX(cx);
        if (draggingIdx >= 0) {
            const lo = draggingIdx > 0 ? seams[draggingIdx - 1] : 0;
            const hi = draggingIdx < seams.length - 1 ? seams[draggingIdx + 1] : paneW;
            seams[draggingIdx] = canvasX.clamp(lo + 5, hi - 5);
            if (e.preventDefault) e.preventDefault();
        } else {
            // hover cursor
            canvas.style.cursor = nearestSeam(canvasX) >= 0 ? "ew-resize" : "default";
        }
    }
    function onUp() { draggingIdx = -1; }

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", onUp);
    canvas.addEventListener("touchstart", function (e) { onDown(e); e.preventDefault(); });
    canvas.addEventListener("touchmove", function (e) { onMove(e); e.preventDefault(); });
    canvas.addEventListener("touchend", onUp);

    function drawLabel(text, midX) {
        if (!text) return;
        ctx.font = "bold " + Math.max(12, Math.round(0.025 * H)) + "px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0,0,0,0.65)";
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.strokeText(text, midX, 8);
        ctx.fillText(text, midX, 8);
    }

    function drawSeam(x) {
        // Vertical line
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 4;
        ctx.stroke();

        // Knob
        const r = 0.045 * H;
        const ky = H / 2;
        ctx.beginPath();
        ctx.arc(x, ky, r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fill();
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 2;
        ctx.stroke();
        // Arrows inside knob
        ctx.fillStyle = "#333";
        const a = r * 0.55;
        const b = r * 0.35;
        ctx.beginPath();
        ctx.moveTo(x - a, ky);
        ctx.lineTo(x - b, ky - b * 0.65);
        ctx.lineTo(x - b, ky + b * 0.65);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x + a, ky);
        ctx.lineTo(x + b, ky - b * 0.65);
        ctx.lineTo(x + b, ky + b * 0.65);
        ctx.closePath();
        ctx.fill();
    }

    function drawLoop() {
        for (let i = 0; i < nPanes; i++) {
            const segStart = i === 0 ? 0 : seams[i - 1];
            const segEnd = i === nPanes - 1 ? paneW : seams[i];
            const w = segEnd - segStart;
            if (w <= 0) continue;
            ctx.drawImage(
                vid,
                i * paneW + segStart, 0, w, H, // src rect inside the hstacked video
                segStart, 0, w, H               // dst rect inside the canvas
            );
        }
        for (let i = 0; i < nPanes; i++) {
            const segStart = i === 0 ? 0 : seams[i - 1];
            const segEnd = i === nPanes - 1 ? paneW : seams[i];
            drawLabel(labels[i] || "", (segStart + segEnd) / 2);
        }
        for (let i = 0; i < seams.length; i++) {
            drawSeam(seams[i]);
        }
        requestAnimationFrame(drawLoop);
    }
    requestAnimationFrame(drawLoop);
}

function resizeAndPlayMulti(element) {
    const nPanes = parseInt(element.dataset.panes || "2", 10);
    const cv = document.getElementById(element.id + "Merge");
    cv.width = element.videoWidth / nPanes;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px"; // hide source video; canvas does the drawing
    playMultiVids(element.id);
}
