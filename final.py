# -*- coding: utf-8 -*-
import os
# 一定要放在 import ultralytics 之前，才能關掉 connectivity check
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from paddleocr import PaddleOCR

from torchvision import models, transforms
from PIL import Image


# ========================== 船類分類：預設(備援)類別名稱 ==========================
# 如果你的 checkpoint 裡面有 class_names，會自動用 checkpoint 的，不用管這裡
# 如果沒有 class_names，就會用這裡當備援（長度不夠會自動補 classX）
SHIP_CLASSES_FALLBACK = [
    "class0", "class1", "class2", "class3",
    "class4", "class5", "class6", "class7"
]


# ========================== OCR（多版本前處理 + auto挑最佳）==========================
ALLOWED_DIGITS = "0123456789"
ALLOWED_ALNUM  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)

def normalize_spaces(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", "").replace("\t", "").replace(" ", "")
    return s.strip()

def postprocess_text(s, digits_only=True, keep_cjk=False):
    """
    - digits_only=True：只保留數字（並 O->0）
    - keep_cjk=True：中文模式，不做英數白名單過濾
    - keep_cjk=False：英數模式，套用 ALLOWED_ALNUM
    """
    if not s:
        return ""

    s = normalize_spaces(s)

    if digits_only:
        s = s.upper().replace("O", "0")
        return "".join([c for c in s if c in ALLOWED_DIGITS])

    if keep_cjk:
        return s

    s = s.upper()
    return "".join([c for c in s if c in ALLOWED_ALNUM])

def preprocess_variants(bgr):
    """產生多種前處理版本，讓 OCR 選最佳結果"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 7, 50, 50)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)

    def binarize(x):
        x = cv2.GaussianBlur(x, (3, 3), 0)
        th = cv2.adaptiveThreshold(
            x, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
        return th

    th1 = binarize(tophat)
    th2 = binarize(blackhat)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, k2, iterations=1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, k2, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, k2, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k2, iterations=1)

    def pad(img, p=12):
        return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=255)

    variants = [
        pad(g),
        pad(255 - g),
        pad(th1),
        pad(255 - th1),
        pad(th2),
        pad(255 - th2),
    ]
    return variants

def extract_texts_from_predict_output(res):
    """兼容 PaddleOCR predict() 多種格式，盡量抓到文字"""
    texts = []
    if isinstance(res, list):
        for item in res:
            if isinstance(item, dict):
                if "rec_texts" in item and isinstance(item["rec_texts"], list):
                    texts.extend(item["rec_texts"])
                if "text" in item and isinstance(item["text"], str):
                    texts.append(item["text"])
            elif isinstance(item, (tuple, list)):
                for sub in item:
                    if isinstance(sub, (tuple, list)) and len(sub) >= 1 and isinstance(sub[0], str):
                        texts.append(sub[0])
    elif isinstance(res, dict):
        if "rec_texts" in res and isinstance(res["rec_texts"], list):
            texts.extend(res["rec_texts"])
        if "text" in res and isinstance(res["text"], str):
            texts.append(res["text"])
    texts = [t for t in texts if t is not None and str(t).strip() != ""]
    return texts

def score_text_general(s: str, prefer_cjk: bool) -> float:
    if not s:
        return -1.0
    s = normalize_spaces(s)
    if not s:
        return -1.0

    cjk_cnt = sum(1 for c in s if is_cjk(c))
    alnum_cnt = sum(1 for c in s.upper() if c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    hyphen_cnt = s.count("-")

    base = len(s)
    eff = alnum_cnt * 0.6 + cjk_cnt * 1.2 + hyphen_cnt * 0.2

    bonus = 0.0
    if prefer_cjk and cjk_cnt > 0:
        bonus += 2.0
    if (not prefer_cjk) and alnum_cnt > 0:
        bonus += 1.0
    return base + eff + bonus

def ocr_best_of_variants(ocr, bgr, digits_only=True, keep_cjk=False, prefer_cjk=False):
    variants = preprocess_variants(bgr)
    best_txt, best_sc = "", -1.0

    for img in variants:
        inp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        res = ocr.predict(inp)

        txts = extract_texts_from_predict_output(res)
        merged = "".join(txts)
        merged = postprocess_text(merged, digits_only=digits_only, keep_cjk=keep_cjk)

        if digits_only:
            sc = len(merged) if merged.isdigit() else (len(merged) - 2)
        else:
            sc = score_text_general(merged, prefer_cjk=prefer_cjk)

        if sc > best_sc:
            best_sc = sc
            best_txt = merged

    return best_txt, best_sc

def build_ocr_instances():
    ocr_en  = PaddleOCR(lang="en", use_textline_orientation=True)
    ocr_cht = PaddleOCR(lang="chinese_cht", use_textline_orientation=True)
    return ocr_en, ocr_cht

def ocr_auto(bgr, ocr_en, ocr_cht, digits_only=False):
    txt_en, sc_en = ocr_best_of_variants(
        ocr_en, bgr,
        digits_only=digits_only,
        keep_cjk=False,
        prefer_cjk=False
    )
    txt_cht, sc_cht = ocr_best_of_variants(
        ocr_cht, bgr,
        digits_only=digits_only,
        keep_cjk=True,
        prefer_cjk=True
    )
    if sc_cht >= sc_en:
        return "cht", txt_cht, sc_cht
    else:
        return "en", txt_en, sc_en


# ========================== 視覺化工具 ==========================
def draw_label(img, x1, y1, text, scale=0.6, thickness=2):
    """在畫面上畫帶底色的文字（可讀性更好）"""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x1 + w + 6)
    y2 = min(img.shape[0] - 1, y1 + h + baseline + 6)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 + h + 3),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

def safe_crop(img, x1, y1, x2, y2, pad=2):
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


# ========================== YOLO 載入（偵測用 / 若分類是 Ultralytics classify 也可用）==========================
def load_ultralytics_yolo(weights_path: str, task_hint: str = "unknown"):
    """
    載入 Ultralytics YOLO 權重。
    如果你丟進來的是非 Ultralytics 的 .pt（例如 Inception state_dict），會在 YOLO() 時爆 KeyError: 'model'
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[找不到權重檔] {weights_path}")
    return YOLO(weights_path)


# ========================== 船類分類：PyTorch InceptionV3（支援 6/8/任意類自動適配）=========================
def _strip_module_prefix(state_dict: dict) -> dict:
    """處理 DataParallel 存的 'module.' 前綴"""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

_TORCH_CLS_TFMS = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _need_aux_logits(state_dict: dict) -> bool:
    """若 checkpoint 含 AuxLogits.* 權重，代表訓練時可能 aux_logits=True"""
    try:
        return any(isinstance(k, str) and k.startswith("AuxLogits.") for k in state_dict.keys())
    except Exception:
        return False


def build_torch_inception_v3(num_classes: int, device: str, aux_logits: bool):
    """
    torchvision InceptionV3（不載預訓練），並把 fc / AuxLogits.fc 換成 num_classes
    - aux_logits=True：forward 會回傳 InceptionOutputs(logits, aux_logits)
    """
    model = models.inception_v3(weights=None, aux_logits=aux_logits, init_weights=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if aux_logits and hasattr(model, "AuxLogits") and model.AuxLogits is not None:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    model.to(device)
    model.eval()
    return model

def _pick_state_dict_from_ckpt(ckpt: dict) -> dict:
    """從 checkpoint dict 裡挑出真正的 state_dict"""
    for k in ["model_state", "state_dict", "model_state_dict", "net", "model"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt  

def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    """從 fc.weight 推斷類別數"""
    if "fc.weight" not in state_dict:
        raise ValueError("[分類權重缺少 fc.weight] 無法推斷類別數，請確認你的 checkpoint 內容。")
    return int(state_dict["fc.weight"].shape[0])

def _build_classes_list(num_classes: int, ckpt: dict, fallback: list) -> list:
    """優先用 ckpt['class_names']，否則用 fallback，不夠就補 classX"""
    if isinstance(ckpt, dict) and "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
        classes = list(ckpt["class_names"])
    else:
        classes = list(fallback)

    classes = classes[:num_classes]
    while len(classes) < num_classes:
        classes.append(f"class{len(classes)}")
    return classes

def load_torch_inception_classifier(weights_path: str, fallback_classes: list, device: str):
    """
    載入 PyTorch InceptionV3 checkpoint，並「自動適配」checkpoint 類別數（6/8/任意）。
    回傳：(model, classes_list)
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[找不到分類權重] {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        model = ckpt.to(device).eval()
        return model, list(fallback_classes)

    state_dict = _pick_state_dict_from_ckpt(ckpt)
    state_dict = _strip_module_prefix(state_dict)

    num_classes = _infer_num_classes_from_state_dict(state_dict)
    classes = _build_classes_list(num_classes, ckpt, fallback_classes)

    aux_logits = _need_aux_logits(state_dict)

    model = build_torch_inception_v3(num_classes=num_classes, device=device, aux_logits=aux_logits)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] InceptionV3 missing keys（前10）：", missing[:10])
    if unexpected:
        print("[WARN] InceptionV3 unexpected keys（前10）：", unexpected[:10])

    model.eval()
    return model, classes

def classify_ship_torch(model, frame_bgr, classes: list):
    """PyTorch InceptionV3 分類：回傳 top1 類別名稱與信心"""
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = _TORCH_CLS_TFMS(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        probs = F.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        name = classes[idx] if 0 <= idx < len(classes) else str(idx)
        return name, conf

def classify_ship_ultralytics(model_cls, frame):
    """Ultralytics classify：回傳 top1 類別名稱與信心"""
    res = model_cls.predict(source=frame, verbose=False)[0]
    top1_idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    name = res.names[top1_idx] if hasattr(res, "names") and res.names else str(top1_idx)
    return name, conf

def load_ship_classifier(ship_cls_path: str):
    """
    自動判斷船類分類模型來源：
    - 若是 Ultralytics classify best.pt：用 YOLO(...) 走 res.probs
    - 否則（例如你現在的 inception_v3/best.pt）：用 PyTorch InceptionV3 推論（自動適配類別數）
    回傳：(backend, payload)
      backend: 'ultralytics' or 'torch'
      payload:
        - ultralytics: YOLO model
        - torch: (torch_model, classes_list)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 先試 Ultralytics classify（若不是 Ultralytics 格式會爆 KeyError: 'model'）
    try:
        m = load_ultralytics_yolo(ship_cls_path, task_hint="classify")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = m.predict(source=dummy, verbose=False)[0]  # 測一下結構
        return "ultralytics", m
    except Exception as e:
        # 你之前看到的：原因： 'model' 就是這裡來的
        print("[INFO] 船類分類權重不是 Ultralytics classify best.pt，改用 PyTorch InceptionV3。原因：", e)

    torch_model, classes = load_torch_inception_classifier(
        ship_cls_path,
        fallback_classes=SHIP_CLASSES_FALLBACK,
        device=device
    )
    return "torch", (torch_model, classes)


# ========================== 船舷號偵測 ==========================
def detect_hull(model_det, frame, conf_thres=0.25):
    """
    YOLOv8 detect：回傳 bbox list: (x1,y1,x2,y2,conf,cls_id,cls_name)
    """
    out = []
    r = model_det.predict(source=frame, conf=conf_thres, verbose=False)[0]
    names = r.names if hasattr(r, "names") else None
    if r.boxes is None:
        return out

    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        cf = float(b.conf[0])
        cid = int(b.cls[0]) if b.cls is not None else 0
        cname = names[cid] if names else str(cid)
        out.append((int(x1), int(y1), int(x2), int(y2), cf, cid, cname))
    return out


# ========================== 影片整合推論：船類分類 + 船舷號偵測 + 船舷號 OCR + 繪製到影片 ==========================
def run_video(
    video_path,
    ship_cls_pt,
    hull_det_pt,
    out_video_path,
    conf_det=0.25,
    ocr_mode="auto",       # auto / en / cht
    digits_only=False,     # True=只要純數字（例如 618/1108）
    frame_stride=1,        # >1 可加速（例如 2 表示每2幀跑一次）
    show_preview=False
):
    # 1) 載入模型
    cls_backend, cls_payload = load_ship_classifier(ship_cls_pt)          # 船類分類（自動判斷 / 自動適配類別數）
    model_det = load_ultralytics_yolo(hull_det_pt, task_hint="detect")    # 船舷號偵測（必須是 YOLO detect best.pt）

    # 2) OCR 初始化（只做一次，避免很慢）
    ocr_en, ocr_cht = build_ocr_instances()

    # 3) 讀影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4) 輸出影片
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    frame_id = 0
    last_ship_name, last_ship_conf = "", 0.0
    last_ocr_results = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        do_infer = (frame_id % frame_stride == 0)

        if do_infer:
            # A) 船類分類
            if cls_backend == "ultralytics":
                model_cls = cls_payload
                ship_name, ship_conf = classify_ship_ultralytics(model_cls, frame)
            else:
                torch_model, torch_classes = cls_payload
                ship_name, ship_conf = classify_ship_torch(torch_model, frame, torch_classes)

            last_ship_name, last_ship_conf = ship_name, ship_conf

            # B) 船舷號偵測
            dets = detect_hull(model_det, frame, conf_thres=conf_det)

            # C) 對每個偵測框做 OCR
            ocr_results = []
            for (x1, y1, x2, y2, cf, cid, cname) in dets:
                crop = safe_crop(frame, x1, y1, x2, y2, pad=3)
                if crop is None:
                    continue

                if ocr_mode == "en":
                    txt, sc = ocr_best_of_variants(
                        ocr_en, crop,
                        digits_only=digits_only,
                        keep_cjk=False,
                        prefer_cjk=False
                    )
                    picked = "en"
                elif ocr_mode == "cht":
                    txt, sc = ocr_best_of_variants(
                        ocr_cht, crop,
                        digits_only=digits_only,
                        keep_cjk=True,
                        prefer_cjk=True
                    )
                    picked = "cht"
                else:
                    picked, txt, sc = ocr_auto(crop, ocr_en, ocr_cht, digits_only=digits_only)

                # 避免全 0 / O 的無效結果
                if txt and len(set(txt)) == 1 and txt[0] in "0O":
                    txt = ""
                    sc = -1

                ocr_results.append((x1, y1, x2, y2, cf, picked, txt, sc))

            last_ocr_results = ocr_results

        # ---------- 繪圖（即使不 infer 也用 last_* 顯示） ----------
        ship_text = f"ShipType: {last_ship_name} ({last_ship_conf:.2f})"
        draw_label(frame, 10, 10, ship_text, scale=0.7, thickness=2)

        for (x1, y1, x2, y2, cf, picked, txt, sc) in last_ocr_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if txt:
                label = f"Hull: {txt} | det {cf:.2f} | {picked} sc {sc:.1f}"
            else:
                label = f"Hull: (none) | det {cf:.2f}"
            draw_label(frame, x1, max(0, y1 - 28), label, scale=0.6, thickness=2)

        writer.write(frame)

        if show_preview:
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print("Done! Output:", out_video_path)


# ========================== Main ==========================
if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\video\video_1.mp4"

    # 船類分類：你目前是 InceptionV3 的 best.pt（這裡會自動用 PyTorch 載入，並自動適配 6/8 類）
    SHIP_CLS_PT = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\ships_dataset\result\Various_categories_8_50\inception_v3\best.pt"

    # 船舷號偵測：必須是 Ultralytics YOLOv8 detect 訓練輸出的 best.pt
    HULL_DET_PT = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\yoloV8\best.pt"

    OUT_VIDEO = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\result\video_1.mp4"  # 存取名稱可更改

    run_video(
        video_path=VIDEO_PATH,
        ship_cls_pt=SHIP_CLS_PT,
        hull_det_pt=HULL_DET_PT,
        out_video_path=OUT_VIDEO,
        conf_det=0.25,
        ocr_mode="auto",     # auto / en / cht
        digits_only=False,   # 若舷號一定純數字才改 True
        frame_stride=1,      # 太慢可改 2 或 3
        show_preview=False   # True 可即時看畫面（ESC退出）
    )
