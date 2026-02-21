自駕船影像感知系統：船型分類 × 船舷號偵測 × OCR 辨識

An integrated vision system for autonomous ships: vessel classification, hull number detection, and OCR.



![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)
![ONNX](https://img.shields.io/badge/ONNX-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)





Project Overview｜專案簡介


本專案為一套自駕船影像感知系統，整合船型分類、船舷號偵測與 OCR 文字辨識，支援影片與即時串流。
系統結合 InceptionV3（船型分類）、YOLOv8n（船舷號偵測）與 PaddleOCR（船舷號辨識），並將結果即時顯示於畫面中。

This project implements an autonomous ship vision system integrating vessel classification, hull number detection, and OCR recognition for video streams and recorded videos.
The system combines InceptionV3 (classification), YOLOv8n (detection), and PaddleOCR (OCR), with real-time visualization.



Pipeline｜系統流程


影片輸入 → 船型分類 → 船舷號偵測 → OCR 辨識 → 結果疊加顯示

Video input → Vessel classification → Hull number detection → OCR recognition → Visualization overlay

輸入（影片 / 即時串流）
        │
        ├─ 船型分類（InceptionV3 / ResNet / EfficientNet）
        │        └─ 輸出：船型類別（Vessel Type）
        │
        ├─ 船舷號偵測（YOLOv8n）
        │        └─ 輸出：船舷號位置框（Bounding Boxes）
        │
        ├─ ROI 裁切 → OCR（PaddleOCR）
        │        └─ 輸出：船舷號碼（Alphanumeric / 可擴充中文）
        │
        └─ 視覺化疊加（Overlay）
                 └─ 輸出：標註後影片 / 即時顯示


Input (Video / Live Stream)
        │
        ├─ Vessel Classification (InceptionV3 / ResNet / EfficientNet)
        │        └─ Output: Vessel Type
        │
        ├─ Hull Number Detection (YOLOv8n)
        │        └─ Output: Bounding Boxes
        │
        ├─ ROI Crop → OCR (PaddleOCR)
        │        └─ Output: Hull Number Text (Alphanumeric, extendable to Chinese)
        │
        └─ Visualization Overlay
                 └─ Output: Annotated Video / Live Display





Features｜功能


船型分類（InceptionV3，可替換 ResNet / EfficientNet等...辨識模型）

船舷號位置偵測（YOLOv8n）

船舷號 OCR（PaddleOCR，支援英文 + 數字，可擴充中文）

影片 / 即時串流即時標註


Features:

Vessel classification (InceptionV3, replaceable with ResNet/EfficientNet)

Hull number detection (YOLOv8n)

OCR recognition (PaddleOCR, English + digits, extendable to Chinese)

Real-time video and stream visualization





Training｜模型訓練簡述

船型分類：以多類船舶影像訓練 InceptionV3，輸出 best.pt / best.onnx

船舷號偵測：使用 YOLOv8n 訓練船舷號框

OCR：PaddleOCR + 自建英數字資料集


Classification: trained InceptionV3, exported best.pt / best.onnx

Detection: trained YOLOv8n for hull numbers

OCR: PaddleOCR with custom alphanumeric dataset





Demo｜影片測試


每一幀顯示：船型、船舷號框、OCR 辨識結果

Each frame displays vessel type, hull bounding box, and OCR result.



Structure｜專案結構
Autonomous_Ship/
    ├── final.py          # 主程式（整合分類 + 偵測 + OCR）
    ├── weights/          # 模型權重
    │   ├── cls_best.pt
    │   └── yolo_best.pt
    ├── OCR_test.py
    └── output/
        Run｜執行方式
    python final.py --video path/to/video.mp4




Future Work｜未來方向

多船追蹤（Multi-object Tracking）

夜間 / 惡劣天候強化

與 AIS、雷達資料融合

整合自駕船決策模組
