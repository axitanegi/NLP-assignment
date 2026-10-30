from ultralytics import YOLO
import cv2
import numpy as np

# ---------- settings ----------
INPUT_VIDEO = "input.mp4"          # your input video
OUTPUT_VIDEO = "pose_black.mp4"    # output video (black background + white COCO points)
MODEL_PATH = "yolo26n-pose.pt"     # same model as teacher's code
CONF = 0.25                        # person detection confidence
KPT_CONF = 0.3                     # minimum keypoint confidence to draw
DRAW_LINES = True                  # set False to show only the points
# ------------------------------


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

WHITE = (255, 255, 255)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_no = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, conf=CONF, verbose=False)

    # black canvas, same size as the video (original frame is discarded)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    kpts = results[0].keypoints
    if kpts is not None and kpts.xy is not None:
        xy_all = kpts.xy.cpu().numpy()                      # (people, 17, 2)
        conf_all = (kpts.conf.cpu().numpy()
                    if kpts.conf is not None
                    else np.ones(xy_all.shape[:2]))         # (people, 17)

        for xy, kc in zip(xy_all, conf_all):
            visible = [(kc[i] >= KPT_CONF and xy[i][0] > 0 and xy[i][1] > 0)
                       for i in range(len(xy))]

            if DRAW_LINES:
                for a, b in SKELETON:
                    if visible[a] and visible[b]:
                        pa = (int(xy[a][0]), int(xy[a][1]))
                        pb = (int(xy[b][0]), int(xy[b][1]))
                        cv2.line(canvas, pa, pb, WHITE, 2, cv2.LINE_AA)

            for i, (x, y) in enumerate(xy):
                if visible[i]:
                    cv2.circle(canvas, (int(x), int(y)), 5, WHITE, -1, cv2.LINE_AA)

    out.write(canvas)
    frame_no += 1

cap.release()
out.release()
print(f"Done - processed {frame_no} frames, saved to {OUTPUT_VIDEO}")
