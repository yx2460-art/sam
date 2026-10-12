import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# =========================
# 1. 路径配置
# =========================
IMAGE_ID = "30"

image_path = (
    f"/home/xubo/xy/bsinet/BsiNet-torch-main_ori/data/test/image/{IMAGE_ID}.tif"
)

instance_path = (
    f"/home/xubo/xy/bsinet/BsiNet-torch-main_ori/save10/instance/{IMAGE_ID}.png"
)

checkpoint = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

SAVE_ROOT = "/home/xubo/xy/sam/segment-anything/sam_refine_save"


DIR_INSTANCE = os.path.join(SAVE_ROOT, "instance")
DIR_INSTANCE_COLOR = os.path.join(SAVE_ROOT, "instance_color")
DIR_OVERLAY = os.path.join(SAVE_ROOT, "overlay")
DIR_CLEAN_MASK = os.path.join(SAVE_ROOT, "clean_mask")
DIR_DEBUG = os.path.join(SAVE_ROOT, "debug")

for d in [
    DIR_INSTANCE,
    DIR_INSTANCE_COLOR,
    DIR_OVERLAY,
    DIR_CLEAN_MASK,
    DIR_DEBUG,
]:
    os.makedirs(d, exist_ok=True)


# =========================
# 2. 读取图像
# =========================
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(image_path)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

instance = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
if instance is None:
    raise FileNotFoundError(instance_path)

if len(instance.shape) == 3:
    instance = instance[:, :, 0]

h, w = image_bgr.shape[:2]

if instance.shape[:2] != (h, w):
    print("警告：原始影像和实例图尺寸不一致，正在 resize instance")
    instance = cv2.resize(instance, (w, h), interpolation=cv2.INTER_NEAREST)

print("原始影像尺寸:", image_bgr.shape)
print("实例图尺寸:", instance.shape)
print("实例编号:", np.unique(instance))


# =========================
# 3. 加载 SAM
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)


# =========================
# 4. 全局变量
# =========================
selected_label = None
current_box = None
current_mask = None
clean_mask = None
refined_instance = instance.copy()

points = []
labels = []

adaptive_params = {
    "box_scale": 1.03,
    "box_min_pad": 2,
    "allowed_kernel": 15,
    "conflict_kernel": 3,
    "smooth_kernel": 3
}


# =========================
# 5. 自适应参数函数
# =========================
def make_odd(x):
    x = int(round(x))
    if x < 1:
        x = 1
    if x % 2 == 0:
        x += 1
    return x


def get_instance_box(inst, lab):
    ys, xs = np.where(inst == lab)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def compute_adaptive_params(box):
    x1, y1, x2, y2 = box
    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)

    short_side = min(obj_w, obj_h)
    long_side = max(obj_w, obj_h)

    # 小田块少外扩，大田块稍微多外扩
    box_scale = 1.00
    box_min_pad = 0

    allowed_kernel = make_odd(np.clip(short_side * 0.12, 3, 15))
    conflict_kernel = make_odd(np.clip(short_side * 0.08, 3, 5))
    smooth_kernel = make_odd(np.clip(short_side * 0.04, 1, 3))

    return {
        "box_scale": box_scale,
        "box_min_pad": box_min_pad,
        "allowed_kernel": allowed_kernel,
        "conflict_kernel": conflict_kernel,
        "smooth_kernel": smooth_kernel,
        "obj_w": obj_w,
        "obj_h": obj_h,
        "short_side": short_side,
        "long_side": long_side
    }


def expand_box(box, img_w, img_h, scale, min_pad):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = x2 - x1
    bh = y2 - y1

    new_w = max(bw * scale, bw + 2 * min_pad)
    new_h = max(bh * scale, bh + 2 * min_pad)

    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(img_w - 1, int(cx + new_w / 2))
    ny2 = min(img_h - 1, int(cy + new_h / 2))

    return np.array([nx1, ny1, nx2, ny2])


# =========================
# 6. 后处理函数
# =========================
def keep_point_connected_component(mask, fg_point):
    mask_uint8 = mask.astype(np.uint8)

    num_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    px, py = fg_point
    target_cc = cc[py, px]

    return cc == target_cc


def regularize_mask(mask):
    global adaptive_params

    smooth_kernel = adaptive_params["smooth_kernel"]

    if smooth_kernel <= 1:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)

    # 先保留最大连通域
    num_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + np.argmax(areas)
        mask_u8 = (cc == max_id).astype(np.uint8)

    kernel = np.ones((smooth_kernel, smooth_kernel), np.uint8)

    # 温和闭运算 + 开运算
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 再保留最大连通域
    num_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels <= 1:
        return mask_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_id = 1 + np.argmax(areas)

    return cc == max_id


def make_instance_overlay(image_bgr, inst):
    overlay = image_bgr.copy()

    ids = np.unique(inst)
    ids = ids[ids != 0]

    for lab in ids:
        obj = (inst == lab).astype(np.uint8)
        contours, _ = cv2.findContours(
            obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return overlay


# =========================
# 7. 赋色
# =========================

def colorize_instance(inst):
    color = np.zeros((inst.shape[0], inst.shape[1], 3), dtype=np.uint8)

    labels_unique = np.unique(inst)
    labels_unique = labels_unique[labels_unique != 0]

    np.random.seed(42)

    for lab in labels_unique:
        c = np.random.randint(50, 255, size=3)
        color[inst == lab] = c

    return color


# =========================
# 8. 显示函数
# =========================
def show_result():
    vis = image_bgr.copy()

    if selected_label is not None:
        selected_region = instance == selected_label
        vis[selected_region] = (
            vis[selected_region] * 0.5 + np.array([255, 255, 0]) * 0.5
        )

    if current_mask is not None:
        vis[current_mask] = (
            vis[current_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
        )

    if current_box is not None:
        x1, y1, x2, y2 = current_box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for p, l in zip(points, labels):
        color = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(vis, tuple(p), 3, color, -1)

    cv2.imshow("adaptive SAM refine", vis)


# =========================
# 9. SAM 推理
# =========================
def run_sam():
    global current_mask, clean_mask

    if selected_label is None:
        print("请先左键选择目标实例")
        return

    input_point = np.array(points)
    input_label = np.array(labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=current_box,
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    current_mask = masks[best_idx]

    print("SAM score:", scores[best_idx])

    fg_point = points[0]
    clean_mask = keep_point_connected_component(current_mask, fg_point)

    selected_region = (instance == selected_label).astype(np.uint8)

    allowed_kernel = adaptive_params["allowed_kernel"]
    kernel = np.ones((allowed_kernel, allowed_kernel), np.uint8)

    allowed_region = cv2.dilate(
        selected_region,
        kernel,
        iterations=1
    ).astype(bool)

    clean_mask = clean_mask & allowed_region

    # 面积约束：防止 SAM 结果比原 BsiNet 实例大太多
    orig_area = np.sum(selected_region > 0)
    new_area = np.sum(clean_mask > 0)

    if new_area > orig_area * 1.08:
        print("SAM 结果面积过大，限制在原实例附近")
        small_kernel = np.ones((3, 3), np.uint8)
        small_allowed = cv2.dilate(
            selected_region,
            small_kernel,
            iterations=1
        ).astype(bool)
        clean_mask = clean_mask & small_allowed

    clean_mask = regularize_mask(clean_mask)

    show_result()


# =========================
# 10. 鼠标交互
# =========================
def mouse_click(event, x, y, flags, param):
    global selected_label, current_box, points, labels, adaptive_params

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 1:
            print("前景点已经有了。如果要重选，请按 r 重置")
            return

        lab = int(instance[y, x])

        if lab == 0:
            print("你点到了背景，请左键点击目标田块内部")
            return

        selected_label = lab
        raw_box = get_instance_box(instance, selected_label)

        adaptive_params = compute_adaptive_params(raw_box)

        current_box = expand_box(
            raw_box,
            w,
            h,
            scale=adaptive_params["box_scale"],
            min_pad=adaptive_params["box_min_pad"]
        )

        points.append([x, y])
        labels.append(1)

        print("前景点:", x, y)
        print("选中的 BsiNet 实例编号:", selected_label)
        print("原始 box:", raw_box)
        print("放大 box:", current_box)
        print("自适应参数:", adaptive_params)

        run_sam()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if selected_label is None:
            print("请先左键点击目标田块")
            return

        if len(labels) >= 2:
            print("背景点已经有了。如果要重选，请按 r 重置")
            return

        points.append([x, y])
        labels.append(0)

        print("背景点:", x, y)

        run_sam()


# =========================
# 11. 保存结果
# =========================
def save_refined_result():
    global refined_instance

    if selected_label is None or clean_mask is None:
        print("还没有有效修正结果，无法保存")
        return

    refined_instance = instance.copy()

    conflict_kernel_size = adaptive_params["conflict_kernel"]
    conflict_kernel = np.ones(
        (conflict_kernel_size, conflict_kernel_size),
        np.uint8
    )

    conflict_region = cv2.dilate(
        clean_mask.astype(np.uint8),
        conflict_kernel,
        iterations=1
    ).astype(bool)

    selected_region = (instance == selected_label).astype(np.uint8)

    allowed_kernel_size = adaptive_params["allowed_kernel"]
    allowed_kernel = np.ones(
        (allowed_kernel_size, allowed_kernel_size),
        np.uint8
    )

    allowed_region = cv2.dilate(
        selected_region,
        allowed_kernel,
        iterations=1
    ).astype(bool)

    conflict_region = conflict_region & allowed_region

    refined_instance[conflict_region] = 0
    refined_instance[instance == selected_label] = 0
    refined_instance[clean_mask] = selected_label

    refined_instance_path = os.path.join(DIR_INSTANCE, f"{IMAGE_ID}.png")
    refined_color_path = os.path.join(DIR_INSTANCE_COLOR, f"{IMAGE_ID}.png")
    refined_overlay_path = os.path.join(DIR_OVERLAY, f"{IMAGE_ID}.png")
    clean_mask_path = os.path.join(DIR_CLEAN_MASK, f"{IMAGE_ID}.png")

    cv2.imwrite(refined_instance_path, refined_instance)
    cv2.imwrite(clean_mask_path, (clean_mask * 255).astype(np.uint8))

    color_inst = colorize_instance(refined_instance)
    cv2.imwrite(refined_color_path, color_inst)

    overlay = make_instance_overlay(image_bgr, refined_instance)
    cv2.imwrite(refined_overlay_path, overlay)


    cv2.namedWindow("clean SAM mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("refined overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("refined instance", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("clean SAM mask", 600, 600)
    cv2.resizeWindow("refined overlay", 800, 800)
    cv2.resizeWindow("refined instance", 600, 600)

    cv2.imshow("clean SAM mask", (clean_mask * 255).astype(np.uint8))
    cv2.imshow("refined overlay", overlay)
    cv2.imshow("refined instance", refined_instance)

    print("已保存:")
    print(refined_instance_path)
    print(refined_color_path)
    print(refined_overlay_path)
    print(clean_mask_path)


# =========================
# 12. 重置
# =========================
def reset():
    global selected_label, current_box, current_mask, clean_mask
    global refined_instance, points, labels, adaptive_params

    selected_label = None
    current_box = None
    current_mask = None
    clean_mask = None
    refined_instance = instance.copy()
    points = []
    labels = []

    adaptive_params = {
        "box_scale": 1.03,
        "box_min_pad": 2,
        "allowed_kernel": 15,
        "conflict_kernel": 3,
        "smooth_kernel": 3
    }

    print("已重置")
    show_result()


# =========================
# 13. 主程序
# =========================
cv2.namedWindow("adaptive SAM refine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("adaptive SAM refine", 1200, 900)

show_result()
cv2.setMouseCallback("adaptive SAM refine", mouse_click)

print("操作说明：")
print("左键一次：目标田块前景点，立即运行 SAM")
print("右键一次：可选背景点，重新运行 SAM")
print("s：保存 refined_instance / refined_overlay / clean_sam_mask")
print("r：重置")
print("q：退出")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_refined_result()

    elif key == ord("r"):
        reset()

    elif key == ord("q"):
        break

cv2.destroyAllWindows()