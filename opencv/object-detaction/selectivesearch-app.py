# -*- coding: utf-8 -*-

import selectivesearch
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""
選擇性搜索範例

selectivesearch.selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50)
im_orig: 輸入圖像
scale: 輸出區域最大個數(表示felzenszwalb分割时，值越大，表示保留的下来的集合就越大)
sigma: felzenszwalb分割算法中高斯分布中的標準差
min_size: felzenszwalb分割算法中區域連通最小個數(表示分割后最小组尺寸)

        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
"""

img = skimage.data.astronaut()
# print(img)
# plt.imshow(img)
# plt.show()


img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

print(regions)

# 計算一共分割了多少個原始候選區域
org_tmp = set()
for i in range(img_lbl.shape[0]):
    for j in range(img_lbl.shape[1]):
        org_tmp.add(img_lbl[i, j, 3])

print('[原始候選區域]: {}'.format(len(org_tmp)))
print('[selective search候選區域]: {}'.format(len(regions)))

candidates = set()
for r in regions:
    if r['rect'] in candidates:
        continue
    # 排除小於2000 pixels的候選區域
    if r['size'] < 2000:
        continue
    # 只保留近似正方形
    x, y, w, h = r['rect']
    if w / h > 1.2 or h / w > 1.2:
        continue
    candidates.add(r['rect'])


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)

for x, y, w, h in candidates:
    print('[selective search候選區域，數值資料]: {} {} {} {}'.format(x, y, w, h))
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

plt.show()




