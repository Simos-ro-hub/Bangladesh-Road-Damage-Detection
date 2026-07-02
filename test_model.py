import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from PIL import Image
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"
DHAKA_CENTER = (23.8103, 90.4125)

CLASS_NAMES = [
    'D00-Longitudinal crack', 'D10-Transverse crack', 'D20-Alligator crack',
    'D30-Repaired crack',     'D40-Pothole',          'D50-Pedestrian crossing blur',
    'D60-Lane line blur',     'D70-Manhole cover',    'D80-Patchy road',
    'D90-Rutting',
]

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
    0: {'minor':  600, 'moderate': 1800, 'severe':  4500},
    1: {'minor':  600, 'moderate': 1800, 'severe':  4500},
    2: {'minor': 1500, 'moderate': 4000, 'severe':  9000},
    3: {'minor':  400, 'moderate': 1000, 'severe':  2500},
    4: {'minor': 1200, 'moderate': 3000, 'severe':  7000},
    5: {'minor':  800, 'moderate': 2000, 'severe':  4000},
    6: {'minor':  500, 'moderate': 1200, 'severe':  2800},
    7: {'minor': 2000, 'moderate': 5000, 'severe': 12000},=
    8: {'minor': 1000, 'moderate': 2500, 'severe':  6000},
    9: {'minor': 1500, 'moderate': 4000, 'severe':  8500},
}

SEV_BGR = {'minor': (53, 200, 53), 'moderate': (30, 160, 220), 'severe': (40, 40, 220)}
PX_TO_M = 7.0 / 640.0
PX2_TO_M2 = PX_TO_M ** 2


# ── Core functions ────────────────────────────────────────────────────────────

def get_severity(area_px, conf, cls_id):
    t = SEVERITY_THRESHOLDS.get(cls_id, (2000, 8000, 0.45, 0.70))
    ag = 1 if area_px < t[0] else 2 if area_px < t[1] else 3
    cg = 1 if conf    < t[2] else 2 if conf    < t[3] else 3
    score = max(1, min(3, round(ag * 0.6 + cg * 0.4)))
    return {1: 'minor', 2: 'moderate', 3: 'severe'}[score], score


def estimate_cost(w, h, cls_id, severity):
    area_m2  = w * h * PX2_TO_M2
    rate     = COST_RATES.get(cls_id, COST_RATES[4])
    cost_bdt = round(area_m2 * rate.get(severity, rate['moderate']), 2)
    return round(area_m2, 4), cost_bdt


def get_gps(img_path):
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        exif = Image.open(img_path)._getexif()
        if exif:
            for tid, val in exif.items():
                if TAGS.get(tid) == 'GPSInfo':
                    gps = {GPSTAGS.get(t, t): v for t, v in val.items()}
                    lat = gps['GPSLatitude']
                    lon = gps['GPSLongitude']
                    lat = lat[0] + lat[1] / 60 + lat[2] / 3600
                    lon = lon[0] + lon[1] / 60 + lon[2] / 3600
                    return round(float(lat), 6), round(float(lon), 6)
    except Exception:
        pass
    return (
        round(DHAKA_CENTER[0] + random.uniform(-0.1, 0.1), 6),
        round(DHAKA_CENTER[1] + random.uniform(-0.1, 0.1), 6),
    )


def detect(frame, model, conf_thr=0.35):
    h, w = frame.shape[:2]
    result = model.predict(frame, conf=conf_thr, iou=0.55,
                           max_det=100, verbose=False)[0]
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

        cls_name  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'cls{cls_id}'
        sev, sev_score = get_severity((x2-x1)*(y2-y1), conf, cls_id)
        area_m2, cost  = estimate_cost(x2-x1, y2-y1, cls_id, sev)

        col = SEV_BGR[sev]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        label = '{} {} {:.2f} BDT{:,.0f}'.format(
            cls_name.split('-')[0], sev.upper()[:3], conf, cost)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        yb = max(y1 - 1, th + 5)
        cv2.rectangle(vis, (x1, yb - th - 4), (x1 + tw + 4, yb), col, -1)
        cv2.putText(vis, label, (x1 + 2, yb - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        dets.append({
            'cls_id': cls_id, 'cls_name': cls_name,
            'conf': round(conf, 4), 'severity': sev,
            'sev_score': sev_score, 'area_m2': area_m2, 'cost_bdt': cost,
        })

    if dets:
        total = sum(d['cost_bdt'] for d in dets)
        for i, txt in enumerate([
            f'Det: {len(dets)}  Cost: BDT {total:,.0f}',
            f'Sev: {sum(1 for d in dets if d["severity"]=="severe")}S '
            f'{sum(1 for d in dets if d["severity"]=="moderate")}M '
            f'{sum(1 for d in dets if d["severity"]=="minor")}m',
        ]):
            cv2.putText(vis, txt, (10, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3)
            cv2.putText(vis, txt, (10, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1)
    return dets, vis


def save_heatmap(records, out_path):
    if not records:
        return
    m = folium.Map(location=list(DHAKA_CENTER), zoom_start=13,
                   tiles='CartoDB positron')
    HeatMap(
        [[r['lat'], r['lon'], r['sev_score']] for r in records],
        radius=22, blur=16,
        gradient={0.2: 'blue', 0.5: 'yellow', 0.8: 'orange', 1.0: 'red'},
        name='Damage Heatmap'
    ).add_to(m)

    color_map = {'minor': 'green', 'moderate': 'orange', 'severe': 'red'}
    mg = folium.FeatureGroup(name='Detections', show=False)
    for r in records:
        sc = color_map.get(r['severity'], 'gray')
        popup = ('<div style="font-family:Arial;font-size:12px;width:210px">'
                 '<b>{}</b><br>Severity: <b style="color:{}">{}</b><br>'
                 'Conf: {:.3f} | Area: {:.4f} m2<br>'
                 'Cost: <b>BDT {:,.0f}</b></div>').format(
            r['cls_name'], sc, r['severity'].upper(),
            r['conf'], r['area_m2'], r['cost_bdt'])
        folium.CircleMarker(
            [r['lat'], r['lon']], radius=6,
            color=sc, fill=True, fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=230),
            tooltip='{} | {} | BDT {:,.0f}'.format(
                r['cls_name'].split('-')[0], r['severity'], r['cost_bdt'])
        ).add_to(mg)
    mg.add_to(m)

    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
              'background:white;padding:12px;border-radius:8px;'
              'border:1px solid #ccc;font-family:Arial;font-size:12px">'
              '<b>Severity</b><br>'
              '<span style="color:green">&#9679;</span> Minor<br>'
              '<span style="color:orange">&#9679;</span> Moderate<br>'
              '<span style="color:red">&#9679;</span> Severe</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    m.save(str(out_path))
    print(f'GPS heatmap saved -> {out_path}')


# ── Source handlers ───────────────────────────────────────────────────────────

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def process_image(img_path, model, out_dir, show, all_records):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f'Cannot read: {img_path}'); return

    dets, vis = detect(frame, model)
    out_path  = out_dir / ('det_' + Path(img_path).name)
    cv2.imwrite(str(out_path), vis)

    lat, lon = get_gps(img_path)
    for d in dets:
        all_records.append({**d, 'lat': lat, 'lon': lon,
                             'source': Path(img_path).name})

    total = sum(d['cost_bdt'] for d in dets)
    print(f'  {Path(img_path).name:40s} {len(dets)} det | BDT {total:,.0f}')
    if show:
        cv2.imshow('Road Damage Detection', vis)
        cv2.waitKey(0)


def process_video(vid_src, model, out_dir, show, skip_n, all_records):
    cap = cv2.VideoCapture(vid_src if vid_src != '0' else 0)
    if not cap.isOpened():
        print(f'Cannot open: {vid_src}'); return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_cam = (str(vid_src) == '0')

    out_vid = None
    if not is_cam:
        out_path = out_dir / ('det_' + Path(str(vid_src)).stem + '.mp4')
        out_vid  = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

    print(f'Video: {w_vid}x{h_vid} | {fps:.0f}fps | {total} frames')
    print('Press Q to quit.\n')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_n == 0:
            dets, vis = detect(frame, model)
            if dets and frame_idx % 30 == 0:
                lat = round(DHAKA_CENTER[0] + random.uniform(-0.05, 0.05), 6)
                lon = round(DHAKA_CENTER[1] + random.uniform(-0.05, 0.05), 6)
                for d in dets:
                    all_records.append({**d, 'lat': lat, 'lon': lon,
                                        'source': f'frame_{frame_idx}'})
        else:
            vis = frame

        if out_vid:
            out_vid.write(vis)
        if show or is_cam:
            cv2.imshow('Road Damage [Q=quit]', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if out_vid:
        out_vid.release()
        print(f'\nAnnotated video -> {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='N-RDD2024 Road Damage Detection — Local Test')
    parser.add_argument('--source', required=True,
                        help='Image, video, folder, or 0 for webcam')
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--output',  default='output')
    parser.add_argument('--conf',    type=float, default=0.35)
    parser.add_argument('--show',    action='store_true',
                        help='Display results in window')
    parser.add_argument('--skip',    type=int, default=2,
                        help='Process every N frames for video (default: 2)')
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f'ERROR: {args.weights} not found.')
        print('Place best.pt in the same folder as this script.')
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model: {args.weights}')
    model = YOLO(args.weights)
    print(f'Classes: {len(CLASS_NAMES)} | Conf threshold: {args.conf}\n')

    all_records = []
    src = args.source
    src_path = Path(src)

    if src == '0':
        process_video('0', model, out_dir, True, args.skip, all_records)

    elif src_path.is_file():
        ext = src_path.suffix.lower()
        if ext in IMG_EXT:
            print(f'Image: {src}\n')
            process_image(src_path, model, out_dir, args.show, all_records)
        elif ext in VID_EXT:
            process_video(src, model, out_dir, args.show, args.skip, all_records)
        else:
            print(f'Unsupported: {ext}')

    elif src_path.is_dir():
        imgs = []
        for ext in IMG_EXT:
            imgs += list(src_path.glob(f'*{ext}'))
        imgs = sorted(set(imgs))
        print(f'Folder: {len(imgs)} images\n')
        for ip in imgs:
            process_image(ip, model, out_dir, args.show, all_records)

    else:
        print(f'Source not found: {src}'); return

    cv2.destroyAllWindows()

    # Save outputs
    if all_records:
        df = pd.DataFrame(all_records)
        csv_path = out_dir / 'detections.csv'
        df.to_csv(csv_path, index=False)

        print(f'\n{"="*50}')
        print(f'Total detections : {len(df):,}')
        print(f'Severe           : {(df["severity"]=="severe").sum():,}')
        print(f'Moderate         : {(df["severity"]=="moderate").sum():,}')
        print(f'Minor            : {(df["severity"]=="minor").sum():,}')
        print(f'Total cost (BDT) : {df["cost_bdt"].sum():,.0f}')
        print(f'CSV saved        -> {csv_path}')

        save_heatmap(all_records, out_dir / 'heatmap.html')
        print(f'All outputs      -> {out_dir}/')
        print('='*50)
    else:
        print('\nNo detections found.')


if __name__ == '__main__':
    main()
