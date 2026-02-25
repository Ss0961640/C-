自駕船影像感知系統：船型分類 × 船舷號偵測 × OCR 辨識

An integrated vision system for autonomous ships: vessel classification, hull number detection, and OCR.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)
![ONNX](https://img.shields.io/badge/ONNX-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)




自駕船影像感知系統
船型分類 × 船舷號偵測 × OCR 辨識

專案簡介
本專案為一套即時自駕船影像感知系統，整合船型分類、船舷號目標偵測與 OCR 文字辨識，建立從影像擷取、AI 推論到結果融合與即時視覺化的完整端到端流程，支援影片與即時串流應用。
系統採用多模型協同架構，結合 InceptionV3（船型分類）、YOLOv8n（船舷號偵測）與 PaddleOCR（船舷號辨識），可於複雜海事場景中即時輸出船隻類型、船舷位置與船舷號碼資訊，作為自駕船感知模組與決策系統之前端視覺輸入。

開發期間
自駕船系統開發：2026/01 – 目前仍在進行中
規劃並實作一套可實際部署於海事場景之自駕船影像感知系統，具備可擴充模型架構與即時推論能力。

系統流程
影片 / 串流輸入
→ 船隻目標偵測（YOLOv8n）
→ 船型分類（InceptionV3）
→ 船舷號區域擷取（YOLOv8n）
→ OCR 辨識（PaddleOCR）
→ 即時視覺化疊加顯示

系統功能

船型分類

使用 InceptionV3 進行船型辨識

可替換 ResNet50 / ResNet101 / EfficientNet 等模型

船舷號目標偵測

使用 YOLOv8n 進行即時定位船舷號區域

船舷號 OCR 辨識

使用 PaddleOCR 進行文字辨識

目前支援英文 + 數字，可擴充中文

影片與即時串流顯示

畫面即時疊加船型、船舷號框選區域與 OCR 結果

模型部署彈性

支援 PyTorch（.pt）與 ONNX（.onnx）模型格式

核心模組設計

一、船隻偵測與定位模組
本模組結合 YOLOv8n 進行即時船隻目標偵測與定位，並搭配 InceptionV3 進行船型分類，可於多船同時出現之複雜海事場景下，穩定輸出船隻位置與類別資訊。
功能重點：

多目標即時偵測

船型分類與空間定位整合

架構可擴充多目標追蹤（MOT）

二、船舷號辨識系統
本模組使用 YOLOv8n 偵測船舷號所在區域，並將偵測到的 ROI 送入 PaddleOCR 進行文字解析與辨識，可即時擷取船舷號碼資訊。
目前支援英文與數字，後續可擴充至中文船名或船籍資訊辨識需求。
功能重點：

船舷號自動擷取

即時 OCR 辨識

可應用於船隻身分識別與海事監控

模型訓練簡述

船型分類模型

以多類船舶影像訓練 InceptionV3

輸出模型格式：best.pt / best.onnx

船舷號偵測模型

使用 YOLOv8n 訓練船舷號定位模型

OCR 模組

使用 PaddleOCR，並搭配自建英數字資料集

影片測試展示
每一幀畫面即時顯示：

船型分類結果

船舷號偵測框

OCR 辨識文字結果

未來發展方向

多船追蹤（Multi-Object Tracking, MOT）

夜間與惡劣天候影像強化

與 AIS、雷達資料進行多模態融合

整合自駕船決策與避碰模組

邊緣裝置即時部署（如 Jetson、ONNX Runtime）
