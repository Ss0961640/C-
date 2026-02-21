#數字 + 英文 + 中文
# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

# ===================== 使用者設定 =====================
IMG_DIR = r"C:\Users\Raymond_yang\Desktop\License_Plates\PaddleOCR\Entertainment_Fishing_Vessel"
OUT_CSV = r"C:\Users\Raymond_yang\Desktop\License_Plates\PaddleOCR\Entertainment_Fishing_Vessel_ocr_results.csv"

# 模式：
#   "auto"  : 同時跑 英文(en) + 繁中(cht)，自動挑最佳
#   "en"    : 只跑英文模型（英數舷號）
#   "cht"   : 只跑繁中模型（中文船名）
OCR_MODE = "auto"

# 若你某些資料夾一定是「純數字」，才開 True（例如 618/1108）
DIGITS_ONLY = False

# 英數合法字元（英文模型後處理會用到）
ALLOWED_DIGITS = "0123456789"
ALLOWED_ALNUM  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

# 你可以視需求把常見符號加進去（例如 / .）
# ALLOWED_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-./"
# =====================================================


def preprocess_variants(bgr):
    """產生多種前處理版本，讓 OCR 選最佳結果"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 對比強化 + 去雜訊
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 7, 50, 50)

    # Top-hat / Black-hat 強化字元
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

    # 形態學開/閉運算抑制雜訊與補洞
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, k2, iterations=1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, k2, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, k2, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k2, iterations=1)

    def pad(img, p=12):
        # 白色 padding，避免字貼邊影響辨識
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


def is_cjk(ch: str) -> bool:
    """判斷是否為常見中日韓文字（CJK）"""
    if not ch:
        return False
    code = ord(ch)
    # CJK Unified Ideographs / Ext-A 常用區段（夠用）
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)


def normalize_spaces(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", "").replace("\t", "").replace(" ", "")
    return s.strip()


def postprocess_text(s, digits_only=True, keep_cjk=False):
    """
    後處理：
    - digits_only=True：只保留數字（並 O->0）
    - keep_cjk=True：中文模式，不做英數白名單過濾（避免中文被濾掉）
    - keep_cjk=False：英數模式，套用 ALLOWED_ALNUM
    """
    if not s:
        return ""

    s = normalize_spaces(s)

    if digits_only:
        s = s.upper().replace("O", "0")
        return "".join([c for c in s if c in ALLOWED_DIGITS])

    if keep_cjk:
        # 中文模式：不要用英數白名單過濾
        # 但你仍可在這裡做一些常見修正（可選）
        return s

    # 英數模式
    s = s.upper()
    return "".join([c for c in s if c in ALLOWED_ALNUM])


def extract_texts_from_predict_output(res):
    """
    PaddleOCR 新版 predict() 回傳格式可能不同
    這個函式用「盡量抓到文字」的方式兼容多種格式
    """
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

    # 清理空值
    texts = [t for t in texts if t is not None and str(t).strip() != ""]
    return texts


def score_text_general(s: str, prefer_cjk: bool) -> float:
    """
    打分策略（用於 auto 模式挑最佳）：
    - 長度越長越好
    - 若 prefer_cjk=True（繁中模型），含 CJK 字元會加分
    - 含英數也算有效字
    """
    if not s:
        return -1.0

    s = normalize_spaces(s)
    if not s:
        return -1.0

    cjk_cnt = sum(1 for c in s if is_cjk(c))
    alnum_cnt = sum(1 for c in s.upper() if c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    hyphen_cnt = s.count("-")

    # 基礎：總長度
    base = len(s)

    # 有效字加權
    eff = alnum_cnt * 0.6 + cjk_cnt * 1.2 + hyphen_cnt * 0.2

    # 偏好加成
    bonus = 0.0
    if prefer_cjk and cjk_cnt > 0:
        bonus += 2.0
    if (not prefer_cjk) and alnum_cnt > 0:
        bonus += 1.0

    return base + eff + bonus


def ocr_best_of_variants(ocr, bgr, digits_only=True, keep_cjk=False, prefer_cjk=False):
    """
    對多個前處理版本做 OCR，挑選分數最高者
    - keep_cjk: 中文模式不做英數白名單過濾
    - prefer_cjk: 用於打分偏好（auto 模式）
    """
    variants = preprocess_variants(bgr)

    best_txt, best_sc = "", -1.0

    for img in variants:
        inp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        res = ocr.predict(inp)

        txts = extract_texts_from_predict_output(res)
        merged = "".join(txts)

        merged = postprocess_text(merged, digits_only=digits_only, keep_cjk=keep_cjk)

        # digits_only 的情境：用簡單長度 + 是否符合數字做分數
        if digits_only:
            sc = len(merged) if merged.isdigit() else (len(merged) - 2)
        else:
            sc = score_text_general(merged, prefer_cjk=prefer_cjk)

        if sc > best_sc:
            best_sc = sc
            best_txt = merged

    return best_txt, best_sc


def build_ocr_instances():
    """
    建立 OCR 實例：
    - 英文：en
    - 繁中：chinese_cht
    """
    ocr_en = PaddleOCR(lang="en", use_textline_orientation=True)
    ocr_cht = PaddleOCR(lang="chinese_cht", use_textline_orientation=True)
    return ocr_en, ocr_cht


def ocr_auto(bgr, ocr_en, ocr_cht, digits_only=False):
    """
    auto 模式：
    - 跑 en + cht
    - 分別挑 best_of_variants
    - 再用分數挑總最佳
    """
    # 英文：英數白名單、偏好英數
    txt_en, sc_en = ocr_best_of_variants(
        ocr_en, bgr,
        digits_only=digits_only,
        keep_cjk=False,
        prefer_cjk=False
    )

    # 繁中：不做英數白名單過濾、偏好 CJK
    txt_cht, sc_cht = ocr_best_of_variants(
        ocr_cht, bgr,
        digits_only=digits_only,
        keep_cjk=True,
        prefer_cjk=True
    )

    # 最終選擇：分數較高者
    if sc_cht >= sc_en:
        return "cht", txt_cht, sc_cht, txt_en, sc_en
    else:
        return "en", txt_en, sc_en, txt_cht, sc_cht


def main():
    ocr_en, ocr_cht = build_ocr_instances()

    rows = []
    for fn in sorted(os.listdir(IMG_DIR)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        path = os.path.join(IMG_DIR, fn)
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        if OCR_MODE.lower() == "en":
            pred, sc = ocr_best_of_variants(
                ocr_en, bgr,
                digits_only=DIGITS_ONLY,
                keep_cjk=False,
                prefer_cjk=False
            )
            picked = "en"
            aux_pred, aux_sc = "", -1

        elif OCR_MODE.lower() == "cht":
            pred, sc = ocr_best_of_variants(
                ocr_cht, bgr,
                digits_only=DIGITS_ONLY,
                keep_cjk=True,
                prefer_cjk=True
            )
            picked = "cht"
            aux_pred, aux_sc = "", -1

        else:
            picked, pred, sc, aux_pred, aux_sc = ocr_auto(
                bgr, ocr_en, ocr_cht, digits_only=DIGITS_ONLY
            )

        rows.append({
            "file": fn,
            "picked_model": picked,      # en / cht（auto時顯示最後選到哪個）
            "pred": pred,               # 最終結果
            "score": sc,                # 最終分數
            "other_pred": aux_pred,     # auto時另一個模型的結果（方便你比對）
            "other_score": aux_sc
        })

        print(f"{fn} -> [{picked}] {pred}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\nDone:", OUT_CSV)
    print(df.head(10))


if __name__ == "__main__":
    main()
