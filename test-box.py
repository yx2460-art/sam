import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

image_path = "/home/xubo/xy/sam/segment-anything/assets/field.tif"
checkpoint = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"无法读取图像：{image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cuda")
sam.eval()

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

drawing = False
start_point = None
end_point = None
current_result = image_bgr.copy()


def run_sam_with_box(box):
    global current_result

    masks, scores, logits = predictor.predict(
        box=np.array(box),
        multimask_output=True
    )

    mask = masks[np.argmax(scores)]

    result = image_bgr.copy()
    result[mask] = result[mask] * 0.5 + np.array([0, 0, 255]) * 0.5

    x1, y1, x2, y2 = box
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_result = result

    cv2.imwrite("sam_box_result.png", result)
    cv2.imwrite("sam_box_mask.png", (mask * 255).astype(np.uint8))

    print("Box:", box)
    print("保存完成：sam_box_result.png 和 sam_box_mask.png")


def mouse_event(event, x, y, flags, param):
    global drawing, start_point, end_point, current_result

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)
        temp = image_bgr.copy()
        cv2.rectangle(temp, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("SAM Box Interactive", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        if x2 - x1 < 5 or y2 - y1 < 5:
            print("框太小，请重新画框")
            return

        run_sam_with_box([x1, y1, x2, y2])
        cv2.imshow("SAM Box Interactive", current_result)


cv2.imshow("SAM Box Interactive", image_bgr)
cv2.setMouseCallback("SAM Box Interactive", mouse_event)

print("操作说明：")
print("鼠标左键按住并拖动：画框")
print("松开鼠标：SAM 根据框分割")
print("按 r：重置")
print("按 q：退出")

while True:
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    elif key == ord("r"):
        current_result = image_bgr.copy()
        cv2.imshow("SAM Box Interactive", current_result)
        print("已重置")

cv2.destroyAllWindows()