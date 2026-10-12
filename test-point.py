import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

image_path = "/home/xubo/xy/sam/segment-anything/assets/field.tif"
checkpoint = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cuda")

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

points = []
labels = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        print("前景点：", x, y)
        run_sam()

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        print("背景点：", x, y)
        run_sam()

def run_sam():
    input_point = np.array(points)
    input_label = np.array(labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask = masks[np.argmax(scores)]

    result = image_bgr.copy()
    result[mask] = result[mask] * 0.5 + np.array([0, 0, 255]) * 0.5

    for p, l in zip(points, labels):
        color = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(result, tuple(p), 4, color, -1)

    cv2.imshow("SAM result", result)
    cv2.imwrite("sam_field_result.png", result)
    cv2.imwrite("sam_field_mask.png", (mask * 255).astype(np.uint8))

cv2.imshow("SAM result", image_bgr)
cv2.setMouseCallback("SAM result", mouse_click)

print("左键：前景点")
print("右键：背景点")
print("按 q 退出")

while True:
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()