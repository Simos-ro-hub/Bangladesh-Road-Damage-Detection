# Bangladesh Road Damage Detection

AI-powered BD road inspection system using YOLOv11m — detects 10 road defect 
classes with severity grading, BDT cost estimation, and GPS heatmap.

## Live Demo
[Hugging Face Spaces](https://huggingface.co/Simosro)

## Model Performance
- **Architecture:** YOLOv11m
- **Dataset:** N-RDD2024 (19,095 images, 6 countries)
- **mAP@50:** 0.582
- **Classes:** 10 (D00–D90)
- **Training:** 50 epochs, AdamW, cosine LR

## 10 Defect Classes
| Code | Class |
|------|-------|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D30 | Repaired crack |
| D40 | Pothole |
| D50 | Pedestrian crossing blur |
| D60 | Lane line blur |
| D70 | Manhole cover |
| D80 | Patchy road |
| D90 | Rutting |

## Pipeline
Detection → Severity Grading → BDT Cost Estimation → GPS Heatmap
