import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

image_path = "/home/xubo/xy/sam/segment-anything/assets/field.tif"
checkpoint = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError("图像读取失败")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cuda")
sam.eval()

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

drawing = False
start_point = None
end_point = None

input_box = None

points = []
labels = []

result_image = image_bgr.copy()


def run_sam():

    global result_image

    if input_box is None:
        return

    if len(points) == 0:
        return

    input_point = np.array(points)
    input_label = np.array(labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=np.array(input_box),
        multimask_output=True
    )

    mask = masks[np.argmax(scores)]

    result = image_bgr.copy()

    result[mask] = (
        result[mask] * 0.5 +
        np.array([0, 0, 255]) * 0.5
    )

    x1, y1, x2, y2 = input_box

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    for p, l in zip(points, labels):

        color = (0, 255, 0) if l == 1 else (0, 0, 255)

        cv2.circle(
            result,
            tuple(p),
            4,
            color,
            -1
        )

    result_image = result

    cv2.imshow("SAM Box+Point", result)

    cv2.imwrite(
        "sam_box_point_result.png",
        result
    )

    cv2.imwrite(
        "sam_box_point_mask.png",
        (mask * 255).astype(np.uint8)
    )

    print("保存完成")


def mouse_event(event, x, y, flags, param):

    global drawing
    global start_point
    global end_point
    global input_box
    global result_image

    # 左键拖框
    if event == cv2.EVENT_LBUTTONDOWN:

        if input_box is None:

            drawing = True
            start_point = (x, y)
            end_point = (x, y)

        else:

            points.append([x, y])
            labels.append(1)

            print("前景点:", x, y)

            run_sam()

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        end_point = (x, y)

        temp = image_bgr.copy()

        cv2.rectangle(
            temp,
            start_point,
            end_point,
            (0, 255, 0),
            2
        )

        cv2.imshow("SAM Box+Point", temp)

    elif event == cv2.EVENT_LBUTTONUP and drawing:

        drawing = False

        end_point = (x, y)

        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])

        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        input_box = [x1, y1, x2, y2]

        temp = image_bgr.copy()

        cv2.rectangle(
            temp,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        result_image = temp

        cv2.imshow("SAM Box+Point", temp)

        print("Box:", input_box)

    # 右键背景点
    elif event == cv2.EVENT_RBUTTONDOWN:

        if input_box is not None:

            points.append([x, y])
            labels.append(0)

            print("背景点:", x, y)

            run_sam()


cv2.imshow("SAM Box+Point", image_bgr)

cv2.setMouseCallback(
    "SAM Box+Point",
    mouse_event
)

print("操作说明")
print("第一步：左键拖动绘制Box")
print("第二步：左键添加前景点")
print("第三步：右键添加背景点")
print("按 r 重置")
print("按 q 退出")

while True:

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    elif key == ord("r"):

        input_box = None

        points = []
        labels = []

        result_image = image_bgr.copy()

        cv2.imshow(
            "SAM Box+Point",
            result_image
        )

        print("已重置")

cv2.destroyAllWindows()