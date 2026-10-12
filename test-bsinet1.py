import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# =========================
# 1. 路径配置
# =========================
IMAGE_ID = "30"

image_path = f"/home/xubo/xy/bsinet/BsiNet-torch-main_ori/data/test/image/{IMAGE_ID}.tif"
instance_path = f"/home/xubo/xy/bsinet/BsiNet-torch-main_ori/save10/instance/{IMAGE_ID}.png"

checkpoint = "/home/xubo/xy/sam/segment-anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

SAVE_ROOT = "/home/xubo/xy/sam/segment-anything/sam_refine_save1"

DIR_INSTANCE = os.path.join(SAVE_ROOT, "instance")
DIR_INSTANCE_COLOR = os.path.join(SAVE_ROOT, "instance_color")
DIR_OVERLAY = os.path.join(SAVE_ROOT, "overlay")
DIR_CLEAN_MASK = os.path.join(SAVE_ROOT, "clean_mask")
DIR_DEBUG = os.path.join(SAVE_ROOT, "debug")

for d in [DIR_INSTANCE, DIR_INSTANCE_COLOR, DIR_OVERLAY, DIR_CLEAN_MASK, DIR_DEBUG]:
    os.makedirs(d, exist_ok=True)


# =========================
# 2. 模式参数
# =========================
BOX_MODE = 2

LOCAL_BOX_SHORT = 55
LOCAL_BOX_LONG = 240

DRAWING_BOX = False
DRAW_MODE = False
manual_box_start = None
manual_box_end = None


# =========================
# 3. 读取数据
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
# 4. 加载 SAM
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image_rgb)


# =========================
# 5. 全局变量
# =========================
selected_label = None
current_box = None
current_mask = None
clean_mask = None
refined_instance = instance.copy()

points = []
labels = []


# =========================
# 6. 工具函数
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

    return {
        "smooth_kernel": make_odd(np.clip(short_side * 0.04, 1, 3)),
        "obj_w": obj_w,
        "obj_h": obj_h
    }


def expand_box(box, img_w, img_h, scale=1.00, min_pad=0):
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


def make_click_box(x, y, img_w, img_h, raw_box):
    x1, y1, x2, y2 = raw_box
    obj_w = x2 - x1
    obj_h = y2 - y1

    if obj_w >= obj_h:
        box_w = LOCAL_BOX_LONG
        box_h = LOCAL_BOX_SHORT
        direction = "horizontal"
    else:
        box_w = LOCAL_BOX_SHORT
        box_h = LOCAL_BOX_LONG
        direction = "vertical"

    bx1 = max(0, x - box_w // 2)
    by1 = max(0, y - box_h // 2)
    bx2 = min(img_w - 1, x + box_w // 2)
    by2 = min(img_h - 1, y + box_h // 2)

    print("局部框方向:", direction)
    return np.array([bx1, by1, bx2, by2])


def keep_point_connected_component(mask, fg_point):
    mask_uint8 = mask.astype(np.uint8)
    num_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    px, py = fg_point
    target_cc = cc[py, px]
    return cc == target_cc


def regularize_mask(mask, smooth_kernel):
    if smooth_kernel <= 1:
        return mask.astype(bool)

    mask_u8 = mask.astype(np.uint8)

    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + np.argmax(areas)
        mask_u8 = (cc == max_id).astype(np.uint8)

    kernel = np.ones((smooth_kernel, smooth_kernel), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(
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


def colorize_instance(inst):
    color = np.zeros((inst.shape[0], inst.shape[1], 3), dtype=np.uint8)

    labels_unique = np.unique(inst)
    labels_unique = labels_unique[labels_unique != 0]

    np.random.seed(42)
    for lab in labels_unique:
        c = np.random.randint(50, 255, size=3)
        color[inst == lab] = c

    return color


def smooth_polyline(points_arr, window=9):
    if len(points_arr) < window:
        return points_arr

    if window % 2 == 0:
        window += 1

    half = window // 2
    smooth = []

    for i in range(len(points_arr)):
        s = max(0, i - half)
        e = min(len(points_arr), i + half + 1)
        avg = np.mean(points_arr[s:e], axis=0)
        smooth.append(avg)

    return np.array(smooth, dtype=np.int32)


# =========================
# 7. 显示函数
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

    if manual_box_start is not None and manual_box_end is not None:
        x1, y1 = manual_box_start
        x2, y2 = manual_box_end
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for p, l in zip(points, labels):
        color = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(vis, tuple(p), 4, color, -1)

    mode_text = f"BOX_MODE={BOX_MODE}  1:instance  2:click-auto  3:manual"
    cv2.putText(
        vis,
        mode_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("SAM refine final", vis)


# =========================
# 8. 由 clean_mask 生成曲线切缝
# =========================
def build_cut_band_from_sam(clean_mask, points, labels, old_region, line_width=5):
    neg_points = [p for p, l in zip(points, labels) if l == 0]
    if len(neg_points) == 0:
        print("警告：没有背景点，无法确定切缝方向")
        return None

    neg = np.array(neg_points, dtype=np.float32)
    neg_center = neg.mean(axis=0)

    ys, xs = np.where(clean_mask > 0)
    if len(xs) < 10:
        print("警告：clean_mask 太小")
        return None

    mask_w = xs.max() - xs.min() + 1
    mask_h = ys.max() - ys.min() + 1

    curve_points = []

    if mask_h >= mask_w:
        take_right = neg_center[0] > xs.mean()

        for yy in range(ys.min(), ys.max() + 1):
            row_xs = xs[ys == yy]
            if len(row_xs) == 0:
                continue
            px = row_xs.max() if take_right else row_xs.min()
            curve_points.append([px, yy])

    else:
        take_bottom = neg_center[1] > ys.mean()

        for xx in range(xs.min(), xs.max() + 1):
            col_ys = ys[xs == xx]
            if len(col_ys) == 0:
                continue
            py = col_ys.max() if take_bottom else col_ys.min()
            curve_points.append([xx, py])

    curve_points = np.array(curve_points, dtype=np.int32)

    if len(curve_points) < 5:
        print("警告：曲线点不足")
        return None

    # 关键：不要过度平滑，否则会重新变直
    curve_points = smooth_polyline(curve_points, window=3)

    cut_band = np.zeros((h, w), dtype=np.uint8)

    cv2.polylines(
        cut_band,
        [curve_points.reshape(-1, 1, 2)],
        False,
        1,
        thickness=line_width
    )

    ys_old, xs_old = np.where(old_region)
    old_y_min, old_y_max = ys_old.min(), ys_old.max()
    old_x_min, old_x_max = xs_old.min(), xs_old.max()

    # 关键：按曲线末端斜率延伸，而不是竖直/水平延伸
    if mask_h >= mask_w:
        p_top = curve_points[0]
        p_bottom = curve_points[-1]

        if len(curve_points) >= 6:
            v_top = curve_points[0] - curve_points[5]
            v_bottom = curve_points[-1] - curve_points[-6]
        else:
            v_top = np.array([0, -1])
            v_bottom = np.array([0, 1])

        if v_top[1] == 0:
            v_top[1] = -1
        if v_bottom[1] == 0:
            v_bottom[1] = 1

        scale_top = (old_y_min - p_top[1]) / v_top[1]
        scale_bottom = (old_y_max - p_bottom[1]) / v_bottom[1]

        pt_top = p_top + scale_top * v_top
        pt_bottom = p_bottom + scale_bottom * v_bottom

        pt_top = np.array([
            np.clip(int(pt_top[0]), 0, w - 1),
            np.clip(int(pt_top[1]), 0, h - 1)
        ])

        pt_bottom = np.array([
            np.clip(int(pt_bottom[0]), 0, w - 1),
            np.clip(int(pt_bottom[1]), 0, h - 1)
        ])

        cv2.line(cut_band, tuple(p_top), tuple(pt_top), 1, thickness=line_width)
        cv2.line(cut_band, tuple(p_bottom), tuple(pt_bottom), 1, thickness=line_width)

    else:
        p_left = curve_points[0]
        p_right = curve_points[-1]

        if len(curve_points) >= 6:
            v_left = curve_points[0] - curve_points[5]
            v_right = curve_points[-1] - curve_points[-6]
        else:
            v_left = np.array([-1, 0])
            v_right = np.array([1, 0])

        if v_left[0] == 0:
            v_left[0] = -1
        if v_right[0] == 0:
            v_right[0] = 1

        scale_left = (old_x_min - p_left[0]) / v_left[0]
        scale_right = (old_x_max - p_right[0]) / v_right[0]

        pt_left = p_left + scale_left * v_left
        pt_right = p_right + scale_right * v_right

        pt_left = np.array([
            np.clip(int(pt_left[0]), 0, w - 1),
            np.clip(int(pt_left[1]), 0, h - 1)
        ])

        pt_right = np.array([
            np.clip(int(pt_right[0]), 0, w - 1),
            np.clip(int(pt_right[1]), 0, h - 1)
        ])

        cv2.line(cut_band, tuple(p_left), tuple(pt_left), 1, thickness=line_width)
        cv2.line(cut_band, tuple(p_right), tuple(pt_right), 1, thickness=line_width)

    near_kernel = np.ones((9, 9), np.uint8)
    old_near = cv2.dilate(old_region.astype(np.uint8), near_kernel, iterations=1).astype(bool)

    cut_band = cut_band.astype(bool) & old_near

    return cut_band


# =========================
# 9. SAM 推理
# =========================
def run_sam():
    global current_mask, clean_mask

    if len(points) == 0:
        print("请至少添加一个前景点")
        return

    if current_box is None:
        print("当前没有 box")
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

    fg_points = [p for p, l in zip(points, labels) if l == 1]
    clean_mask = keep_point_connected_component(current_mask, fg_points[0])

    box_mask = np.zeros_like(instance, dtype=bool)
    x1, y1, x2, y2 = current_box.astype(int)
    box_mask[y1:y2 + 1, x1:x2 + 1] = True
    clean_mask = clean_mask & box_mask

    if selected_label is not None:
        raw_box = get_instance_box(instance, selected_label)
        params = compute_adaptive_params(raw_box)
        smooth_kernel = params["smooth_kernel"]
    else:
        smooth_kernel = 1

    clean_mask = regularize_mask(clean_mask, smooth_kernel)

    show_result()


# =========================
# 10. 鼠标交互
# =========================
def mouse_click(event, x, y, flags, param):
    global selected_label, current_box, points, labels
    global DRAWING_BOX, DRAW_MODE, manual_box_start, manual_box_end

    if DRAW_MODE:
        if event == cv2.EVENT_LBUTTONDOWN:
            DRAWING_BOX = True
            manual_box_start = (x, y)
            manual_box_end = (x, y)
            show_result()
            return

        elif event == cv2.EVENT_MOUSEMOVE and DRAWING_BOX:
            manual_box_end = (x, y)
            show_result()
            return

        elif event == cv2.EVENT_LBUTTONUP and DRAWING_BOX:
            DRAWING_BOX = False
            DRAW_MODE = False
            manual_box_end = (x, y)

            x1 = min(manual_box_start[0], manual_box_end[0])
            y1 = min(manual_box_start[1], manual_box_end[1])
            x2 = max(manual_box_start[0], manual_box_end[0])
            y2 = max(manual_box_start[1], manual_box_end[1])

            current_box = np.array([x1, y1, x2, y2])
            print("手动画框完成:", current_box)
            show_result()
            return

    if event == cv2.EVENT_LBUTTONDOWN:
        lab = int(instance[y, x])

        if lab == 0:
            print("你点到了背景，请点击田块内部")
            return

        if selected_label is None:
            selected_label = lab
            raw_box = get_instance_box(instance, selected_label)

            if BOX_MODE == 1:
                current_box = expand_box(raw_box, w, h, scale=1.00, min_pad=0)
                print("模式1：实例外接框")

            elif BOX_MODE == 2:
                current_box = make_click_box(x, y, w, h, raw_box)
                print("模式2：点击局部框，自适应横/竖")

            elif BOX_MODE == 3:
                if current_box is None:
                    print("模式3请先按 b，然后左键拖动画框")
                    return
                print("模式3：手动画框")

            print("选中实例编号:", selected_label)
            print("SAM 使用 box:", current_box)

        points.append([x, y])
        labels.append(1)
        print("添加前景点:", x, y)
        run_sam()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_box is None:
            print("请先确定 box")
            return

        points.append([x, y])
        labels.append(0)
        print("添加背景点:", x, y)
        run_sam()



# =========================
# 11. 规则化
# =========================
def regularize_instance_labels(inst, min_area=30, kernel_size=3):
    out = np.zeros_like(inst)

    ids = np.unique(inst)
    ids = ids[ids != 0]

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for lab in ids:
        mask = (inst == lab).astype(np.uint8)

        # 闭运算：填小缺口
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 开运算：去小毛刺
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[cc == i] = lab

    return out


# =========================
# 12. 保存结果
# =========================
def save_refined_result():
    global refined_instance

    if selected_label is None or clean_mask is None:
        print("还没有有效结果，无法保存")
        return

    refined_instance = instance.astype(np.uint16).copy()
    old_region = (instance == selected_label)

    cut_band = build_cut_band_from_sam(
        clean_mask,
        points,
        labels,
        old_region,
        line_width=5
    )

    if cut_band is None:
        print("切缝生成失败，无法保存")
        return

    split_region = old_region.copy()
    split_region[cut_band & old_region] = False

    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(
        split_region.astype(np.uint8),
        connectivity=8
    )

    refined_instance[old_region] = 0

    used_labels = set(np.unique(refined_instance).tolist())

    def get_unused_label():
        for lab in range(1, 65535):
            if lab not in used_labels:
                used_labels.add(lab)
                return lab
        raise ValueError("没有可用的新实例编号")

    if num_labels <= 2:
        print("警告：切缝后没有真正分成两块")
        refined_instance[old_region] = selected_label

    else:
        fg_x, fg_y = points[0]
        fg_cc = cc[fg_y, fg_x]

        if fg_cc == 0:
            print("警告：前景点不在有效连通域内")
            refined_instance[old_region] = selected_label
        else:
            new_label = get_unused_label()

            # 前景点所在区域作为目标实例
            target_mask = (cc == fg_cc)

            # 把 clean_mask 向下/周围轻微膨胀，用来吸收断裂的小块
            attach_kernel = np.ones((15, 15), np.uint8)
            attach_region = cv2.dilate(target_mask.astype(np.uint8), attach_kernel, iterations=2).astype(bool)

            for i in range(1, num_labels):
                part_mask = (cc == i)
                area = stats[i, cv2.CC_STAT_AREA]

                if i == fg_cc:
                    refined_instance[part_mask] = new_label
                    continue

                # 小碎块默认归到前景点目标田块
                # 因为这些小碎块多数是切缝造成的断裂残片
                if area < 800:
                    refined_instance[part_mask] = new_label
                else:
                    refined_instance[part_mask] = selected_label

    # 规则化实例边界
    refined_instance = regularize_instance_labels(
        refined_instance,
        min_area=30,
        kernel_size=3
    )

    refined_instance_path = os.path.join(DIR_INSTANCE, f"{IMAGE_ID}.png")
    refined_color_path = os.path.join(DIR_INSTANCE_COLOR, f"{IMAGE_ID}.png")
    refined_overlay_path = os.path.join(DIR_OVERLAY, f"{IMAGE_ID}.png")
    clean_mask_path = os.path.join(DIR_CLEAN_MASK, f"{IMAGE_ID}.png")
    cut_path = os.path.join(DIR_DEBUG, f"{IMAGE_ID}_cut_band.png")

    cv2.imwrite(refined_instance_path, refined_instance)
    cv2.imwrite(clean_mask_path, (clean_mask * 255).astype(np.uint8))

    color_inst = colorize_instance(refined_instance)
    cv2.imwrite(refined_color_path, color_inst)

    overlay = make_instance_overlay(image_bgr, refined_instance)
    cv2.imwrite(refined_overlay_path, overlay)

    cut_vis = (cut_band * 255).astype(np.uint8)
    cv2.imwrite(cut_path, cut_vis)

    cv2.namedWindow("clean SAM mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cut band", cv2.WINDOW_NORMAL)
    cv2.namedWindow("refined overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("refined instance color", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("clean SAM mask", 600, 600)
    cv2.resizeWindow("cut band", 600, 600)
    cv2.resizeWindow("refined overlay", 800, 800)
    cv2.resizeWindow("refined instance color", 600, 600)

    cv2.imshow("clean SAM mask", (clean_mask * 255).astype(np.uint8))
    cv2.imshow("cut band", cut_vis)
    cv2.imshow("refined overlay", overlay)
    cv2.imshow("refined instance color", color_inst)

    print("已保存:")
    print(refined_instance_path)
    print(refined_color_path)
    print(refined_overlay_path)
    print(clean_mask_path)
    print(cut_path)
    print("连通域数量 num_labels:", num_labels)


# =========================
# 13. 重置
# =========================
def reset_all():
    global selected_label, current_box, current_mask, clean_mask
    global refined_instance, points, labels
    global DRAWING_BOX, DRAW_MODE, manual_box_start, manual_box_end

    selected_label = None
    current_box = None
    current_mask = None
    clean_mask = None
    refined_instance = instance.copy()
    points = []
    labels = []

    DRAWING_BOX = False
    DRAW_MODE = False
    manual_box_start = None
    manual_box_end = None

    print("已重置")
    show_result()


# =========================
# 13. 主程序
# =========================
cv2.namedWindow("SAM refine final", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SAM refine final", 1200, 900)

show_result()
cv2.setMouseCallback("SAM refine final", mouse_click)

print("操作说明：")
print("1：实例外接框模式")
print("2：点击局部框模式，自动判断横向/竖向")
print("3：手动画框模式")
print("b：开始手动画框")
print("左键：添加前景点")
print("右键：添加背景点")
print("空格：重新运行 SAM")
print("s：保存")
print("r：重置")
print("q：退出")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("1"):
        BOX_MODE = 1
        print("切换到模式1：实例外接框")
        reset_all()

    elif key == ord("2"):
        BOX_MODE = 2
        print("切换到模式2：点击局部框")
        reset_all()

    elif key == ord("3"):
        BOX_MODE = 3
        print("切换到模式3：手动画框")
        reset_all()

    elif key == ord("b"):
        DRAW_MODE = True
        print("开始画框：左键拖动")

    elif key == ord(" "):
        run_sam()

    elif key == ord("s"):
        save_refined_result()

    elif key == ord("r"):
        reset_all()

    elif key == ord("q"):
        break

cv2.destroyAllWindows()