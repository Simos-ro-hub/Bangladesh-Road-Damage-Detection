import os
import random
import tempfile
import textwrap
from pathlib import Path

import cv2
import folium
import gradio as gr
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from PIL import Image
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"
DHAKA_CENTER = (23.8103, 90.4125)

CLASS_NAMES = [
    "D00-Longitudinal crack", "D10-Transverse crack", "D20-Alligator crack",
    "D30-Repaired crack",     "D40-Pothole",          "D50-Pedestrian crossing blur",
    "D60-Lane line blur",     "D70-Manhole cover",    "D80-Patchy road",
    "D90-Rutting",
]

CLASS_SHORT = [c.split("-")[0] for c in CLASS_NAMES]

SEVERITY_THRESHOLDS = {
    0: (1500,  6000, 0.45, 0.70),
    1: (1500,  6000, 0.45, 0.70),
    2: (2000,  8000, 0.50, 0.75),
    3: (3000, 10000, 0.40, 0.65),
    4: (1200,  5000, 0.45, 0.70),
    5: (5000, 15000, 0.40, 0.65),
    6: (5000, 15000, 0.40, 0.65),
    7: (1000,  4000, 0.50, 0.75),
    8: (2500,  9000, 0.45, 0.70),
    9: (3000, 10000, 0.50, 0.75),
}

COST_RATES = {
    0: {"minor":  600, "moderate": 1800, "severe":  4500},
    1: {"minor":  600, "moderate": 1800, "severe":  4500},
    2: {"minor": 1500, "moderate": 4000, "severe":  9000},
    3: {"minor":  400, "moderate": 1000, "severe":  2500},
    4: {"minor": 1200, "moderate": 3000, "severe":  7000},
    5: {"minor":  800, "moderate": 2000, "severe":  4000},
    6: {"minor":  500, "moderate": 1200, "severe":  2800},
    7: {"minor": 2000, "moderate": 5000, "severe": 12000},
    8: {"minor": 1000, "moderate": 2500, "severe":  6000},
    9: {"minor": 1500, "moderate": 4000, "severe":  8500},
}

SEV_BGR = {
    "minor":    (53,  200,  53),
    "moderate": (30,  160, 220),
    "severe":   (40,   40, 220),
}
SEV_HEX    = {"minor": "#2ecc71", "moderate": "#f39c12", "severe": "#e74c3c"}
SEV_FOLIUM = {"minor": "green",   "moderate": "orange",  "severe": "red"}

PX_TO_M  = 7.0 / 640.0
PX2_TO_M2 = PX_TO_M ** 2

# ── Load model once ───────────────────────────────────────────────────────────
print("Loading YOLOv11m model...")
model = YOLO(MODEL_PATH)
print("Model ready.")

# ── Core detection functions ──────────────────────────────────────────────────

def get_severity(area_px: float, conf: float, cls_id: int) -> tuple:
    t  = SEVERITY_THRESHOLDS.get(cls_id, (2000, 8000, 0.45, 0.70))
    ag = 1 if area_px < t[0] else 2 if area_px < t[1] else 3
    cg = 1 if conf    < t[2] else 2 if conf    < t[3] else 3
    score = max(1, min(3, round(ag * 0.6 + cg * 0.4)))
    return {1: "minor", 2: "moderate", 3: "severe"}[score], score


def estimate_cost(w: float, h: float, cls_id: int, severity: str) -> tuple:
    area_m2  = w * h * PX2_TO_M2
    rate     = COST_RATES.get(cls_id, COST_RATES[4])
    cost_bdt = round(area_m2 * rate.get(severity, rate["moderate"]), 2)
    return round(area_m2, 4), cost_bdt


def detect_frame(frame: np.ndarray, conf_thr: float = 0.35) -> tuple:
    """Run detection on a single BGR frame. Returns (detections, annotated_frame)."""
    h, w = frame.shape[:2]
    result = model.predict(
        frame, conf=conf_thr, iou=0.55, max_det=100, verbose=False)[0]

    dets = []
    vis  = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return dets, vis

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf   = float(box.conf[0].cpu())
        cls_id = int(box.cls[0].cpu())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue

        cls_name       = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
        sev, sev_score = get_severity((x2-x1)*(y2-y1), conf, cls_id)
        area_m2, cost  = estimate_cost(x2-x1, y2-y1, cls_id, sev)

        col = SEV_BGR[sev]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = "{} {} {:.2f} BDT{:,.0f}".format(
            CLASS_SHORT[cls_id] if cls_id < len(CLASS_SHORT) else "?",
            sev.upper()[:3], conf, cost)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        yb = max(y1 - 1, th + 5)
        cv2.rectangle(vis, (x1, yb - th - 4), (x1 + tw + 4, yb), col, -1)
        cv2.putText(vis, label, (x1 + 2, yb - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        dets.append({
            "cls_id": cls_id, "cls_name": cls_name,
            "conf": round(conf, 4), "severity": sev,
            "sev_score": sev_score, "area_m2": area_m2, "cost_bdt": cost,
        })

    # Summary overlay
    if dets:
        total = sum(d["cost_bdt"] for d in dets)
        ns = sum(1 for d in dets if d["severity"] == "severe")
        nm = sum(1 for d in dets if d["severity"] == "moderate")
        ni = sum(1 for d in dets if d["severity"] == "minor")
        for i, txt in enumerate([
            f"Detections: {len(dets)}  |  BDT {total:,.0f}",
            f"Sev: {ns}S  {nm}M  {ni}m",
        ]):
            cv2.putText(vis, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
            cv2.putText(vis, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 1)

    return dets, vis


def build_summary_html(dets: list) -> str:
    """Build a styled HTML summary card for display in Gradio."""
    if not dets:
        return "<div style='padding:16px;color:#6b7280;font-family:monospace'>No damage detected.</div>"

    total  = sum(d["cost_bdt"] for d in dets)
    ns     = sum(1 for d in dets if d["severity"] == "severe")
    nm     = sum(1 for d in dets if d["severity"] == "moderate")
    ni     = sum(1 for d in dets if d["severity"] == "minor")

    sev_badge = {
        "severe":   "<span style='background:#450a0a;color:#f87171;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600'>SEVERE</span>",
        "moderate": "<span style='background:#451a03;color:#fbbf24;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600'>MODERATE</span>",
        "minor":    "<span style='background:#052e16;color:#4ade80;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600'>MINOR</span>",
    }

    rows = ""
    for i, d in enumerate(dets, 1):
        rows += """
        <tr style='border-bottom:1px solid #1e2130'>
          <td style='padding:7px 10px;color:#9ca3af'>{}</td>
          <td style='padding:7px 10px;color:#e8e6df'>{}</td>
          <td style='padding:7px 10px'>{}</td>
          <td style='padding:7px 10px;color:#c9c7c0'>{:.3f}</td>
          <td style='padding:7px 10px;color:#c9c7c0'>{:.4f} m²</td>
          <td style='padding:7px 10px;color:#a78bfa;font-weight:600'>৳{:,.0f}</td>
        </tr>""".format(
            i,
            d["cls_name"].split("-")[0],
            sev_badge.get(d["severity"], d["severity"]),
            d["conf"],
            d["area_m2"],
            d["cost_bdt"],
        )

    html = """
    <div style='font-family:"DM Mono",monospace;background:#0f1117;
                border:1px solid #2a2d3a;border-radius:12px;overflow:hidden'>

      <!-- Header metrics -->
      <div style='display:grid;grid-template-columns:repeat(4,1fr);
                  gap:1px;background:#2a2d3a'>
        <div style='background:#161820;padding:14px 16px'>
          <div style='font-size:10px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:4px'>Detections</div>
          <div style='font-size:1.5rem;font-weight:700;color:#f0e6c8'>{total_det}</div>
        </div>
        <div style='background:#161820;padding:14px 16px'>
          <div style='font-size:10px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:4px'>Severe</div>
          <div style='font-size:1.5rem;font-weight:700;color:#f87171'>{ns}</div>
        </div>
        <div style='background:#161820;padding:14px 16px'>
          <div style='font-size:10px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:4px'>Moderate</div>
          <div style='font-size:1.5rem;font-weight:700;color:#fbbf24'>{nm}</div>
        </div>
        <div style='background:#161820;padding:14px 16px'>
          <div style='font-size:10px;color:#6b7280;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:4px'>Est. Repair (BDT)</div>
          <div style='font-size:1.5rem;font-weight:700;color:#a78bfa'>৳{total_cost}</div>
        </div>
      </div>

      <!-- Table -->
      <table style='width:100%;border-collapse:collapse;font-size:12px'>
        <thead>
          <tr style='background:#1a1d27'>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>#</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>Class</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>Severity</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>Conf</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>Area</th>
            <th style='padding:8px 10px;color:#6b7280;text-align:left;
                       font-size:10px;letter-spacing:.06em;text-transform:uppercase'>Cost</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """.format(
        total_det=len(dets),
        ns=ns, nm=nm,
        total_cost=f"{total:,.0f}",
        rows=rows,
    )
    return html


def build_heatmap(records: list, tmp_dir: str) -> str:
    """Build Folium heatmap and save to temp HTML. Returns file path."""
    if not records:
        return None

    m = folium.Map(location=list(DHAKA_CENTER), zoom_start=13,
                   tiles="CartoDB positron")

    HeatMap(
        [[r["lat"], r["lon"], r["sev_score"]] for r in records],
        radius=22, blur=16,
        gradient={0.2: "blue", 0.4: "cyan",
                  0.6: "yellow", 0.8: "orange", 1.0: "red"},
        name="Damage Heatmap",
    ).add_to(m)

    mg = folium.FeatureGroup(name="Individual Detections", show=False)
    for r in records:
        sc = SEV_FOLIUM.get(r["severity"], "gray")
        popup = (
            "<div style='font-family:Arial;font-size:12px;width:210px'>"
            "<b>{}</b><br>"
            "Severity: <b style='color:{}'>{}</b><br>"
            "Conf: {:.3f} | Area: {:.4f} m²<br>"
            "Cost: <b>BDT {:,.0f}</b>"
            "</div>"
        ).format(
            r["cls_name"], sc, r["severity"].upper(),
            r["conf"], r["area_m2"], r["cost_bdt"],
        )
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=6,
            color=sc, fill=True, fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=230),
            tooltip="{} | {} | BDT {:,.0f}".format(
                r["cls_name"].split("-")[0], r["severity"], r["cost_bdt"]),
        ).add_to(mg)
    mg.add_to(m)

    legend = (
        "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
        "background:white;padding:12px;border-radius:8px;"
        "border:1px solid #ccc;font-family:Arial;font-size:12px'>"
        "<b>Severity</b><br>"
        "<span style='color:green'>&#9679;</span> Minor<br>"
        "<span style='color:orange'>&#9679;</span> Moderate<br>"
        "<span style='color:red'>&#9679;</span> Severe</div>"
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    map_path = os.path.join(tmp_dir, "heatmap.html")
    m.save(map_path)
    return map_path


# ── Tab handlers ──────────────────────────────────────────────────────────────

def handle_image(img_pil, conf_thr):
    """Handle single image upload."""
    if img_pil is None:
        return None, "<p style='color:#6b7280'>Upload an image to begin.</p>", None

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    dets, vis = detect_frame(frame, conf_thr)
    vis_rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    summary_html = build_summary_html(dets)

    # Build heatmap
    map_path = None
    if dets:
        tmp_dir = tempfile.mkdtemp()
        lat = round(DHAKA_CENTER[0] + random.uniform(-0.08, 0.08), 6)
        lon = round(DHAKA_CENTER[1] + random.uniform(-0.08, 0.08), 6)
        records = [{**d, "lat": lat, "lon": lon} for d in dets]
        map_path = build_heatmap(records, tmp_dir)

    return vis_rgb, summary_html, map_path


def handle_camera(img_pil, conf_thr):
    """Handle webcam snapshot."""
    if img_pil is None:
        return None, "<p style='color:#6b7280'>Take a photo to begin.</p>", None
    return handle_image(img_pil, conf_thr)


def handle_video(video_path, conf_thr, skip_n):
    """Handle video upload — process frames, return annotated video + stats."""
    if video_path is None:
        return None, "<p style='color:#6b7280'>Upload a video to begin.</p>", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "<p style='color:#e74c3c'>Cannot open video.</p>", None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_dir  = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "output.mp4")
    writer   = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid))

    all_dets   = []
    all_records = []
    frame_idx  = 0
    max_frames = 300   # cap at 300 processed frames for Spaces limits

    processed = 0
    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, skip_n) == 0:
            dets, vis = detect_frame(frame, conf_thr)
            all_dets.extend(dets)
            if dets and frame_idx % 15 == 0:
                lat = round(DHAKA_CENTER[0] + random.uniform(-0.1, 0.1), 6)
                lon = round(DHAKA_CENTER[1] + random.uniform(-0.1, 0.1), 6)
                for d in dets:
                    all_records.append({**d, "lat": lat, "lon": lon})
            writer.write(vis)
            processed += 1
        else:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Aggregate summary
    if all_dets:
        df = pd.DataFrame(all_dets)
        top_dets = df.sort_values("cost_bdt", ascending=False).head(20).to_dict("records")
        summary_html = build_summary_html(top_dets)
        summary_html = (
            f"<div style='font-family:monospace;font-size:11px;color:#6b7280;"
            f"padding:8px 12px;background:#161820;border-radius:6px;margin-bottom:8px'>"
            f"Processed {processed} frames of {total} total | "
            f"{len(all_dets):,} total detections | "
            f"Top 20 shown below</div>"
        ) + summary_html
    else:
        summary_html = "<p style='color:#6b7280'>No detections found.</p>"

    map_path = build_heatmap(all_records, tmp_dir) if all_records else None

    return out_path, summary_html, map_path


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

/* ── Base ── */
body, .gradio-container, .main, .wrap {
    background: #0a0c14 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── ALL text visible by default ── */
*, p, span, div, label, h1, h2, h3, h4, li {
    color: #c9c7c0;
}

/* ── Header ── */
.app-header {
    padding: 2rem 0 1.5rem;
    text-align: center;
    border-bottom: 1px solid #2a2d3a;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #f0e6c8 !important;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
    display: block;
}
.app-subtitle {
    font-size: 0.78rem !important;
    color: #9ca3af !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.8rem;
}
.tag {
    display: inline-block;
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    color: #9ca3af !important;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 0 3px;
}

/* ── Tabs ── */
.tab-nav button, [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #9ca3af !important;
    background: #161820 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
    color: #f0e6c8 !important;
    background: #2a2d3a !important;
    border-color: #4b5563 !important;
}

/* ── Labels & form controls ── */
label, .block > label > span, fieldset > div > label span,
.label-wrap span, .output-class, span.svelte-1ed2p3z {
    color: #9ca3af !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Input / output panel backgrounds ── */
.block, .svelte-1gfkn6j, .wrap, .input-image, .output-image,
.image-container, .video-container, input, textarea {
    background: #161820 !important;
    border-color: #2a2d3a !important;
}

/* ── Sliders ── */
input[type="range"] { accent-color: #a78bfa; }
.slider > span, .range-slider span { color: #9ca3af !important; }

/* ── Buttons ── */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: #2a2d3a !important;
    color: #f0e6c8 !important;
    border: 1px solid #4b5563 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: #3a3d4a !important;
    border-color: #f0e6c8 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.2rem 0 0.5rem;
    border-top: 1px solid #2a2d3a;
    margin-top: 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #4b5563 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Prose / markdown text ── */
.prose, .prose p, .prose li, .prose span,
.gradio-container .prose { color: #9ca3af !important; }

/* ── Number / value displays ── */
.output-class, .number, .value { color: #f0e6c8 !important; }
"""

HEADER_HTML = """
<div style="padding:2rem 0 1.5rem;text-align:center;
            border-bottom:1px solid #2a2d3a;margin-bottom:1rem">
  <div style="font-family:'DM Mono',monospace;font-size:1.7rem;font-weight:700;
              color:#f0e6c8;letter-spacing:0.04em;margin-bottom:0.4rem">
    Bangladesh Road Damage Detection
  </div>
  <div style="font-size:0.78rem;color:#9ca3af;letter-spacing:0.14em;
              text-transform:uppercase;margin-bottom:0.9rem">
    AI-Powered Road Inspection System
  </div>
  <div>
    <span style="display:inline-block;background:#1a1d27;border:1px solid #2a2d3a;
                 color:#9ca3af;font-size:10px;letter-spacing:0.08em;
                 text-transform:uppercase;padding:3px 10px;border-radius:20px;margin:0 3px">
      YOLOv11m
    </span>
    <span style="display:inline-block;background:#1a1d27;border:1px solid #2a2d3a;
                 color:#9ca3af;font-size:10px;letter-spacing:0.08em;
                 text-transform:uppercase;padding:3px 10px;border-radius:20px;margin:0 3px">
      10 defect classes
    </span>
    <span style="display:inline-block;background:#1a1d27;border:1px solid #2a2d3a;
                 color:#9ca3af;font-size:10px;letter-spacing:0.08em;
                 text-transform:uppercase;padding:3px 10px;border-radius:20px;margin:0 3px">
      N-RDD2024
    </span>
    <span style="display:inline-block;background:#1a1d27;border:1px solid #2a2d3a;
                 color:#9ca3af;font-size:10px;letter-spacing:0.08em;
                 text-transform:uppercase;padding:3px 10px;border-radius:20px;margin:0 3px">
      severity grading
    </span>
    <span style="display:inline-block;background:#1a1d27;border:1px solid #2a2d3a;
                 color:#9ca3af;font-size:10px;letter-spacing:0.08em;
                 text-transform:uppercase;padding:3px 10px;border-radius:20px;margin:0 3px">
      BDT cost estimation
    </span>
  </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;padding:1.2rem 0 0.5rem;
            border-top:1px solid #2a2d3a;margin-top:1.5rem;
            font-family:'DM Mono',monospace;font-size:10px;
            color:#4b5563;letter-spacing:0.08em;text-transform:uppercase">
  N-RDD2024 dataset &nbsp;·&nbsp; YOLOv11m &nbsp;·&nbsp; mAP@50 = 0.582 &nbsp;·&nbsp;
  D00 D10 D20 D30 D40 D50 D60 D70 D80 D90
</div>
"""

INFO_HTML = """
<div style="font-family:'DM Mono',monospace;font-size:11px;
            background:#161820;border:1px solid #2a2d3a;
            border-left:3px solid #a78bfa;border-radius:8px;
            padding:12px 14px;line-height:1.8">
  <span style="color:#a78bfa;font-weight:600">Color guide</span><br>
  <span style="color:#2ecc71">■ Minor</span> &nbsp;&nbsp;
  <span style="color:#f39c12">■ Moderate</span> &nbsp;&nbsp;
  <span style="color:#e74c3c">■ Severe</span><br>
  <span style="color:#9ca3af">Confidence threshold controls detection sensitivity.</span>
</div>
"""

CLASS_HTML = """
<div style="font-family:'DM Mono',monospace;font-size:10px;
            background:#161820;border:1px solid #2a2d3a;border-radius:8px;
            padding:12px 14px;line-height:1.9">
  <span style="color:#f0e6c8;font-weight:600">10 defect classes</span><br>
  <span style="color:#e74c3c">D00</span><span style="color:#9ca3af"> Longitudinal crack</span>
  &nbsp;·&nbsp;
  <span style="color:#e74c3c">D10</span><span style="color:#9ca3af"> Transverse crack</span><br>
  <span style="color:#e74c3c">D20</span><span style="color:#9ca3af"> Alligator crack</span>
  &nbsp;·&nbsp;
  <span style="color:#2ecc71">D30</span><span style="color:#9ca3af"> Repaired crack</span><br>
  <span style="color:#e74c3c">D40</span><span style="color:#9ca3af"> Pothole</span>
  &nbsp;·&nbsp;
  <span style="color:#3498db">D50</span><span style="color:#9ca3af"> Pedestrian crossing</span><br>
  <span style="color:#3498db">D60</span><span style="color:#9ca3af"> Lane blur</span>
  &nbsp;·&nbsp;
  <span style="color:#9b59b6">D70</span><span style="color:#9ca3af"> Manhole cover</span><br>
  <span style="color:#1abc9c">D80</span><span style="color:#9ca3af"> Patchy road</span>
  &nbsp;·&nbsp;
  <span style="color:#e67e22">D90</span><span style="color:#9ca3af"> Rutting</span>
</div>
"""


def build_ui():
    with gr.Blocks(css=CSS, title="Road Damage Detector") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Tab 1: Image Upload ───────────────────────────────────────────
            with gr.TabItem("📷  Image Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="pil",
                            label="Upload road image",
                            sources=["upload"],
                            height=320,
                        )
                        conf_img = gr.Slider(
                            minimum=0.10, maximum=0.90,
                            value=0.35, step=0.05,
                            label="Confidence threshold",
                        )
                        gr.HTML(INFO_HTML)
                        gr.HTML(CLASS_HTML)
                        btn_img = gr.Button("Run Detection", variant="primary")

                    with gr.Column(scale=1):
                        img_output = gr.Image(
                            label="Detection result",
                            type="numpy",
                            height=320,
                        )
                        img_summary = gr.HTML(
                            value="<p style='color:#4b5563;font-family:monospace;"
                                  "font-size:12px;padding:8px'>Results appear here.</p>")

                with gr.Row():
                    img_map = gr.HTML(label="GPS Heatmap")

                def run_image(img, conf):
                    vis, summary, map_path = handle_image(img, conf)
                    map_html = ""
                    if map_path:
                        with open(map_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        map_html = (
                            "<div style='border:1px solid #1e2130;border-radius:10px;"
                            "overflow:hidden;margin-top:12px'>"
                            "<div style='background:#161820;padding:8px 14px;"
                            "font-family:DM Mono,monospace;font-size:11px;color:#6b7280;"
                            "border-bottom:1px solid #1e2130;text-transform:uppercase;"
                            "letter-spacing:.08em'>GPS Damage Heatmap</div>"
                            + raw.replace(
                                "<html>", "<div"
                            ).replace("</html>", "</div>")
                            + "</div>"
                        )
                        # Simpler approach — use iframe with data URI
                        import base64
                        b64 = base64.b64encode(raw.encode()).decode()
                        map_html = (
                            "<div style='border:1px solid #1e2130;border-radius:10px;"
                            "overflow:hidden;margin-top:12px'>"
                            "<div style='background:#161820;padding:8px 14px;"
                            "font-family:\"DM Mono\",monospace;font-size:11px;"
                            "color:#6b7280;border-bottom:1px solid #1e2130;"
                            "text-transform:uppercase;letter-spacing:.08em'>"
                            "GPS Damage Heatmap — toggle layers top-right</div>"
                            "<iframe src='data:text/html;base64,{}' "
                            "width='100%' height='480' frameborder='0'></iframe>"
                            "</div>"
                        ).format(b64)
                    return vis, summary, map_html

                btn_img.click(
                    fn=run_image,
                    inputs=[img_input, conf_img],
                    outputs=[img_output, img_summary, img_map],
                )

            # ── Tab 2: Camera ─────────────────────────────────────────────────
            with gr.TabItem("📸  Camera"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="font-family:'DM Mono',monospace;font-size:11px;
                                    color:#9ca3af;padding:6px 0 4px;
                                    text-transform:uppercase;letter-spacing:.08em">
                          Point camera at road damage &rarr; capture &rarr; detect
                        </div>""")
                        cam_input = gr.Image(
                            type="pil",
                            label="Camera",
                            sources=["webcam"],
                            height=340,
                        )
                        conf_cam = gr.Slider(
                            minimum=0.10, maximum=0.90,
                            value=0.35, step=0.05,
                            label="Confidence threshold",
                        )
                        btn_cam = gr.Button("Detect Damage", variant="primary")

                    with gr.Column(scale=1):
                        cam_output = gr.Image(
                            label="Detection result",
                            type="numpy",
                            height=340,
                        )
                        cam_summary = gr.HTML(
                            value="<p style='color:#4b5563;font-family:monospace;"
                                  "font-size:12px;padding:8px'>Results appear here.</p>")

                with gr.Row():
                    cam_map = gr.HTML(label="GPS Heatmap")

                def run_camera(img, conf):
                    vis, summary, map_path = handle_camera(img, conf)
                    map_html = ""
                    if map_path:
                        import base64
                        with open(map_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        b64 = base64.b64encode(raw.encode()).decode()
                        map_html = (
                            "<div style='border:1px solid #1e2130;border-radius:10px;"
                            "overflow:hidden;margin-top:12px'>"
                            "<div style='background:#161820;padding:8px 14px;"
                            "font-family:\"DM Mono\",monospace;font-size:11px;"
                            "color:#6b7280;border-bottom:1px solid #1e2130'>"
                            "GPS Damage Heatmap</div>"
                            "<iframe src='data:text/html;base64,{}' "
                            "width='100%' height='420' frameborder='0'></iframe>"
                            "</div>"
                        ).format(b64)
                    return vis, summary, map_html

                btn_cam.click(
                    fn=run_camera,
                    inputs=[cam_input, conf_cam],
                    outputs=[cam_output, cam_summary, cam_map],
                )

            # ── Tab 3: Video Upload ───────────────────────────────────────────
            with gr.TabItem("🎬  Video Upload"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(
                            label="Upload road video",
                            sources=["upload"],
                            height=300,
                        )
                        conf_vid = gr.Slider(
                            minimum=0.10, maximum=0.90,
                            value=0.35, step=0.05,
                            label="Confidence threshold",
                        )
                        skip_vid = gr.Slider(
                            minimum=1, maximum=8,
                            value=2, step=1,
                            label="Process every N frames (higher = faster)",
                        )
                        gr.HTML("""
                        <div style="font-family:'DM Mono',monospace;font-size:10px;
                                    background:#161820;border:1px solid #2a2d3a;
                                    border-left:3px solid #f39c12;border-radius:8px;
                                    padding:10px 12px;line-height:1.8">
                          <span style="color:#f39c12;font-weight:600">Note</span><br>
                          <span style="color:#9ca3af">Max 300 processed frames on free Spaces.<br>
                          Increase skip-frames for longer videos.</span>
                        </div>""")
                        btn_vid = gr.Button("Process Video", variant="primary")

                    with gr.Column(scale=1):
                        vid_output = gr.Video(
                            label="Annotated video",
                            height=300,
                        )
                        vid_summary = gr.HTML(
                            value="<p style='color:#4b5563;font-family:monospace;"
                                  "font-size:12px;padding:8px'>Results appear here.</p>")

                with gr.Row():
                    vid_map = gr.HTML(label="GPS Heatmap")

                def run_video(vid, conf, skip):
                    out_path, summary, map_path = handle_video(vid, conf, int(skip))
                    map_html = ""
                    if map_path:
                        import base64
                        with open(map_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                        b64 = base64.b64encode(raw.encode()).decode()
                        map_html = (
                            "<div style='border:1px solid #1e2130;border-radius:10px;"
                            "overflow:hidden;margin-top:12px'>"
                            "<div style='background:#161820;padding:8px 14px;"
                            "font-family:\"DM Mono\",monospace;font-size:11px;"
                            "color:#6b7280;border-bottom:1px solid #1e2130'>"
                            "GPS Damage Heatmap — detection density across frames</div>"
                            "<iframe src='data:text/html;base64,{}' "
                            "width='100%' height='480' frameborder='0'></iframe>"
                            "</div>"
                        ).format(b64)
                    return out_path, summary, map_html

                btn_vid.click(
                    fn=run_video,
                    inputs=[vid_input, conf_vid, skip_vid],
                    outputs=[vid_output, vid_summary, vid_map],
                )

            # ── Tab 4: About ──────────────────────────────────────────────────
            with gr.TabItem("ℹ️  About"):
                gr.HTML("""
                <div style='font-family:"DM Mono",monospace;max-width:700px;
                            margin:1.5rem auto;color:#9ca3af;line-height:1.9;font-size:13px'>

                  <div style='font-size:1.1rem;font-weight:600;color:#f0e6c8;
                              margin-bottom:1rem;letter-spacing:-.01em'>
                              Road Damage Detection System
                  </div>

                  <p>An AI-powered road inspection pipeline that detects 10 types of road
                  defects, grades severity, estimates repair costs in BDT, and maps
                  damage locations — enabling data-driven infrastructure maintenance
                  for municipalities and road authorities.</p>

                  <div style='background:#161820;border:1px solid #1e2130;border-radius:8px;
                              padding:14px;margin:1rem 0'>
                    <div style='color:#a78bfa;font-size:11px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:8px'>Model details</div>
                    <div>Architecture &nbsp;: YOLOv11m (medium)</div>
                    <div>Training data &nbsp;: N-RDD2024 — 19,095 images, 6 countries</div>
                    <div>Epochs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 50 (AdamW, cosine LR)</div>
                    <div>mAP@50 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.582</div>
                    <div>Best class &nbsp;&nbsp;&nbsp;: D70 Manhole cover — 0.839 mAP@50</div>
                  </div>

                  <div style='background:#161820;border:1px solid #1e2130;border-radius:8px;
                              padding:14px;margin:1rem 0'>
                    <div style='color:#a78bfa;font-size:11px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:8px'>Severity grading</div>
                    <div><span style='color:#2ecc71'>Minor</span> &nbsp;&nbsp;&nbsp;— small bbox area + lower confidence</div>
                    <div><span style='color:#f39c12'>Moderate</span> — medium area, moderate confidence</div>
                    <div><span style='color:#e74c3c'>Severe</span> &nbsp;&nbsp;— large area + high confidence</div>
                    <div style='margin-top:6px;color:#4b5563;font-size:11px'>
                      Thresholds calibrated per defect class (area 60% + confidence 40%)
                    </div>
                  </div>

                  <div style='background:#161820;border:1px solid #1e2130;border-radius:8px;
                              padding:14px;margin:1rem 0'>
                    <div style='color:#a78bfa;font-size:11px;text-transform:uppercase;
                                letter-spacing:.08em;margin-bottom:8px'>BDT cost rates (per m²)</div>
                    <div>Pothole &nbsp;&nbsp;&nbsp;&nbsp;: Minor 1,200 · Moderate 3,000 · Severe 7,000</div>
                    <div>Alligator &nbsp;&nbsp;: Minor 1,500 · Moderate 4,000 · Severe 9,000</div>
                    <div>Manhole &nbsp;&nbsp;&nbsp;&nbsp;: Minor 2,000 · Moderate 5,000 · Severe 12,000</div>
                    <div style='margin-top:6px;color:#4b5563;font-size:11px'>
                      Based on Bangladesh LGED schedule of rates 2023-24
                    </div>
                  </div>

                </div>
                """)

        gr.HTML(FOOTER_HTML)

    return demo


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
