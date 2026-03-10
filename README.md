# 自駕船系統開發專案：船隻偵測定位 × 船舷號識別 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-orange)
![ONNX](https://img.shields.io/badge/ONNX-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

以深度學習模型整合 YOLOv8n、InceptionV3 與 PaddleOCR，
本專案建構基於電腦視覺的船舶辨識與追蹤系統，提升船舶監測與自動航行決策能力。

---



https://github.com/user-attachments/assets/f79d4de2-7380-4360-a960-b54e18d32bcc




## 專案簡介 (Overview)
整合船型分類、船舷號目標偵測與 OCR 文字辨識，
建立從「影像擷取 → AI 模型推論 → 結果融合 → 即時視覺化」的完整端到端流程，
支援影片與即時串流場景，適用於海事監控與自駕船感知模組前端。
- 多模型協同架構設計（分類 × 偵測 × OCR）
- 即時影像推論與結果融合流程
- 複雜海事場景下之多船辨識與定位
- 模型推論流程之系統化整合與部署設計


## 系統架構 (System Architecture)
- 影像輸入模組（Video / Stream Input）
- 船隻目標偵測（YOLOv8n）
- 船型分類（InceptionV3）
- 船舷號區域擷取（YOLOv8n ROI）
- OCR 文字辨識（PaddleOCR）
- 結果融合與即時視覺化模組

## 影像感知流程 (Perception Pipeline)
- 讀取影片或即時串流影像
- 使用 YOLOv8n 偵測畫面中船隻與船舷號區域
- 將船隻影像送入 InceptionV3 進行船型分類
- 擷取船舷號 ROI 送入 PaddleOCR 進行文字辨識
- 將船型、船舷位置與 OCR 結果即時疊加顯示於畫面中

---

## 模組說明 (Modules)
   ### 船隻偵測與船型分類模組
      - 使用 YOLOv8n 進行多船即時目標偵測
      - 使用 InceptionV3 進行船型分類
      - 支援多船同時出現之影像場景
      - 輸出船隻類別與對應 Bounding Box
   
   ### 船舷號辨識模組
      - 使用 YOLOv8n 偵測船舷號位置
      - 將 ROI 送入 PaddleOCR 進行文字解析
      - 目前支援英文與數字，後續可擴充中文辨識

## 模型訓練與部署 (Training & Deployment)
   ### 船型分類模型
      - 使用多類船舶影像訓練 InceptionV3
      - 輸出模型格式：best.pt / best.onnx
   
   ### 船舷號偵測模型
      - 使用 YOLOv8n 訓練船舷號定位模型
   
   ### OCR 模組
      - PaddleOCR + 自建英數字資料集

## 系統輸出 (Output)
   ### 每一幀影像即時輸出
      - 船型分類結果
      - 船隻與船舷號偵測框
      - OCR 辨識文字結果

---

## 未來發展方向 (Future Work)
- 多船追蹤（Multi-Object Tracking, MOT）
- 夜間與惡劣天候影像強化
- 與 AIS、雷達資料進行多模態融合
- 整合自駕船決策與避碰模組
- 邊緣裝置即時部署（Jetson / ONNX Runtime）
