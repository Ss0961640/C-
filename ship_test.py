import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from collections import deque

# =========================
# 路徑
# =========================
VIDEO_PATH = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\video\video_1.mp4"
CKPT_PATH  = r"C:\Users\Raymond_yang\Desktop\Autonomous_Ship\ships_dataset\result\Various_categories_6\inception_v3\best.pt"

# =========================
# 建立/載入模型（重點：支援 model_state）
# =========================
def load_inception_classifier(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 從 checkpoint 取資訊（你這個 best.pt 內就有）
    class_names = ckpt.get("class_names", None)
    img_size = int(ckpt.get("img_size", 299))
    mean = ckpt.get("mean", (0.485, 0.456, 0.406))
    std  = ckpt.get("std",  (0.229, 0.224, 0.225))

    if class_names is None:
        raise ValueError("checkpoint 內找不到 class_names，請確認 best.pt 格式")

    num_classes = len(class_names)

    # 建立模型結構（要跟你訓練時一致）
    model = models.inception_v3(weights=None, aux_logits=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # 正確取得 state_dict
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # 退而求其次：若 ckpt 本身就是 state_dict
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device).eval()

    # 回傳：模型 + 類別 + 前處理參數
    return model, class_names, img_size, mean, std

# =========================
# 前處理
# =========================
def preprocess_bgr(frame_bgr: np.ndarray, img_size: int, mean, std):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std  = np.array(std, dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))               # CHW
    x = torch.from_numpy(img).unsqueeze(0)           # (1,3,H,W)
    return x

# =========================
# 影片逐幀分類 + 畫字
# =========================
@torch.no_grad()
def classify_video(video_path: str, model, classes, device, img_size, mean, std):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    # 平滑（可先設 1 看效果，避免卡住錯覺）
    smooth_k = 3
    prob_queue = deque(maxlen=smooth_k)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("播放完畢")
            break

        x = preprocess_bgr(frame, img_size, mean, std).to(device, non_blocking=True)

        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_queue.append(probs)
        probs_s = np.mean(np.stack(prob_queue, axis=0), axis=0)

        pred_idx = int(np.argmax(probs_s))
        pred_name = classes[pred_idx]
        conf = float(probs_s[pred_idx])

        # 顯示 top3（你要除錯很有用）
        top3 = np.argsort(-probs_s)[:3]
        top3_text = " | ".join([f"{classes[i]} {probs_s[i]*100:.1f}%" for i in top3])

        cv2.putText(frame, f"Ship Type: {pred_name} ({conf*100:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Top3: {top3_text}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Ship Classification", frame)
        if (cv2.waitKey(delay) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    model, classes, img_size, mean, std = load_inception_classifier(CKPT_PATH, device)
    print("[INFO] classes =", classes)
    classify_video(VIDEO_PATH, model, classes, device, img_size, mean, std)

if __name__ == "__main__":
    main()
