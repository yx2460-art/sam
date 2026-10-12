import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


IMAGE_DIR = "/home/xubo/xy/bsinet/BsiNet-torch-main_ori/data/train-yn/image"
SAVE_DIR = "/home/xubo/xy/bsinet/BsiNet-torch-main_ori/data/train-yn"

CHECKPOINT = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

MASK_DIR = os.path.join(SAVE_DIR, "mask")
INSTANCE_DIR = os.path.join(SAVE_DIR, "instance")
VIS_DIR = os.path.join(SAVE_DIR, "vis")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)


img_names = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".tif")])

idx = 0
image_bgr = None
image_rgb = None
show_img = None
instance = None
points = []
current_label = 1
current_name = None


def color_for_label(label):
    np.random.seed(label * 123)
    return np.random.randint(50, 255, 3).tolist()


def draw_show():
    global show_img

    show_img = image_bgr.copy()

    if instance is not None and instance.max() > 0:
        overlay = show_img.copy()

        for lbl in np.unique(instance):
            if lbl == 0:
                continue

            region = instance == lbl
            color = np.array(color_for_label(int(lbl)), dtype=np.uint8)

            overlay[region] = (
                overlay[region] * 0.45 + color * 0.55
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                region.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

        show_img = overlay

    for i, (x, y) in enumerate(points):
        cv2.circle(show_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            show_img,
            str(i + 1),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.putText(
        show_img,
        f"{current_name} | left click point | p predict | s save | n next | r reset point | q quit",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1
    )

    cv2.imshow("SAM Simple Instance", show_img)


def load_image():
    global image_bgr, image_rgb, instance, points, current_label, current_name

    img_name = img_names[idx]
    current_name = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMAGE_DIR, img_name)
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(img_path)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w = image_bgr.shape[:2]
    instance = np.zeros((h, w), dtype=np.uint16)
    points = []
    current_label = 1

    print("\n当前图像:", img_name, f"({idx + 1}/{len(img_names)})")
    draw_show()


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print("添加点:", x, y)
        draw_show()


def predict_current():
    global points, instance, current_label

    if len(points) == 0:
        print("没有点，无法预测")
        return

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.ones(len(points), dtype=np.int32)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(bool)

    # 不覆盖之前已经标好的实例
    mask = mask & (instance == 0)

    area = int(mask.sum())
    if area < 20:
        print("预测区域太小，跳过")
        points = []
        draw_show()
        return

    instance[mask] = current_label
    print("加入实例:", current_label, "area:", area)

    current_label += 1
    points = []

    draw_show()


def save_result():
    base = current_name

    instance_path = os.path.join(INSTANCE_DIR, base + ".png")
    mask_path = os.path.join(MASK_DIR, base + ".png")
    vis_path = os.path.join(VIS_DIR, base + ".png")

    binary_mask = (instance > 0).astype(np.uint8) * 255

    cv2.imwrite(instance_path, instance.astype(np.uint16))
    cv2.imwrite(mask_path, binary_mask)
    cv2.imwrite(vis_path, show_img)

    print("保存完成:")
    print("instance:", instance_path)
    print("mask:", mask_path)
    print("vis:", vis_path)


cv2.namedWindow("SAM Simple Instance", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SAM Simple Instance", mouse_callback)

if len(img_names) == 0:
    raise RuntimeError("IMAGE_DIR 中没有 tif 图像")

load_image()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("p"):
        predict_current()

    elif key == ord("s"):
        save_result()

    elif key == ord("r"):
        points = []
        print("已清空当前点")
        draw_show()

    elif key == ord("n"):
        idx += 1
        if idx >= len(img_names):
            print("全部图像完成")
            break
        load_image()

    elif key == ord("q"):
        print("退出")
        break

cv2.destroyAllWindows()