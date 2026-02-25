# 自駕船影像感知系統：船型分類 × 船舷號偵測 × OCR 辨識
# An integrated vision system for autonomous ships: vessel classification, hull number detection, and OCR.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)
![ONNX](https://img.shields.io/badge/ONNX-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

以深度學習模型整合 YOLOv8n、InceptionV3 與 PaddleOCR，
建立即時自駕船影像感知與船舷號辨識系統。

### 專案簡介 (Overview)
本專案建構一套即時自駕船影像感知系統，整合船型分類、船舷號目標偵測與 OCR 文字辨識，
建立從「影像擷取 → AI 模型推論 → 結果融合 → 即時視覺化」的完整端到端流程，
支援影片與即時串流場景，適用於海事監控與自駕船感知模組前端。

- 多模型協同架構設計（分類 × 偵測 × OCR）
- 即時影像推論與結果融合流程
- 複雜海事場景下之多船辨識與定位
- 模型推論流程之系統化整合與部署設計
