import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 假设你的 DataFrame 长这样
df = pd.DataFrame({
    'index'     : [0],
    'question'  : ['guy on left of screen red shirt'],
    # 'height'    : [640],
    # 'width'     : [374],
    'height'    : [480],
    'width'     : [640],
    # 这里我们假设 answer 和 prediction b[都是 list/tuple
    # 'answer'    : [[0, 147, 62, 482]],
    'answer': [[238, 348, 343, 541]],
    # 'prediction': [[  0,   152,   163,   476]]
    'prediction': [[493, 536, 718, 849]]
# [0, 231, 169, 748]]
})



# 读图并转成 RGB
# img_path = "/mnt/public/usr/sunzhichao/benchmark/images/debug/28.jpg"
img_path = "/mnt/public/usr/sunzhichao/benchmark/images/finetune_refcoco_testA/1087.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 原始尺寸
orig_h = df.loc[0,'height']
orig_w = df.loc[0,'width']



# 2) 计算缩放比例
scale_x =  orig_w / 1000
scale_y =  orig_h / 1000

# # 3) resize 图像
# img_rs = cv2.resize(img, (new_w, new_h))

# 4) 提取 GT bbox 并缩放
x1, y1, x2, y2 = df.loc[0, 'answer']
# new_x1 = int(round(x1 * scale_x))
# new_y1 = int(round(y1 * scale_y))
# new_x2 = int(round(x2 * scale_x))
# new_y2 = int(round(y2 * scale_y))
new_x1 = x1
new_y1 = y1

new_x2 = x2
new_y2 = y2

# # 5) 提取 prediction bbox 并缩放
px1, py1, px2, py2 = df.loc[0, 'prediction']
new_px1 = int(round(px1 * scale_x))
new_py1 = int(round(py1 * scale_y))
new_px2 = int(round(px2 * scale_x))
new_py2 = int(round(py2 * scale_y))

# 6) 在 img_rs 上画框
#    cv2.rectangle( img, 左上 pt, 右下 pt, color(BGR), thickness )
#    注意我们的 img_rs 是 RGB，所以传给 cv2 时要用 RGB
vis = img.copy()

# 红色画 GT
cv2.rectangle(vis,
              (new_x1, new_y1),
              (new_x2, new_y2),
              color=(255, 0, 0),  # 红 (R,G,B)
              thickness=3)

cv2.putText(vis, "GT",
            (new_x1, new_y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color=(255,0,0),
            thickness=2)

# 绿色画 Prediction
cv2.rectangle(vis,
              (new_px1, new_py1),
              (new_px2, new_py2),
              color=(0, 255, 0),  # 绿
              thickness=3)

cv2.putText(vis, "Pred",
            (new_px1, new_py1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color=(0,255,0),
            thickness=2)

# 7) 用 matplotlib 显示
plt.figure(figsize=(8,6))
plt.imshow(vis)
plt.axis('off')
plt.title("red=GT(red box), green=Pred(green box)")

output_path = "/mnt/public/usr/sunzhichao/DeepSeek-VL2/try_files/2_try_debug_deepseek_28.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
