# Copyright (c) chenjie04. All rights reserved.

from re import I
from telnetlib import SUPPRESS_LOCAL_ECHO
from matplotlib import pyplot as plt
import numpy as np
from pycocotools.coco import COCO

train_anno_file = "../data/DUO/annotations/instances_train.json"

coco = COCO(train_anno_file)

imgIDS = coco.getImgIds()

print("\nTotal images: ", len(imgIDS))

sum_img_ids = 0
num_objects = 0
small_objects = 0
medium_objects = 0
large_objects = 0
num_holothurian_s = 0
num_holothurian_m = 0
num_holothurian_l = 0
num_echinus_s = 0
num_echinus_m = 0
num_echinus_l = 0
num_scallop_s = 0
num_scallop_m = 0
num_scallop_l = 0
num_starfish_s = 0
num_starfish_m = 0
num_starfish_l = 0

# "categories": [{"name": "holothurian", "id": 1}, {"name": "echinus", "id": 2}, {"name": "scallop", "id": 3}, {"name": "starfish", "id": 4}]}

for id in imgIDS:
    img_info = coco.loadImgs(id)[0]
    height = img_info["height"]
    width = img_info["width"]
    # print("\nheight, width: ", height, width)
    sum_img_ids += 1

    ann_ids = coco.getAnnIds(imgIds=id)
    # print("\nann_ids = ", ann_ids)

    ann_info = coco.loadAnns(ann_ids)
    for _, ann in enumerate(ann_info):
        num_objects += 1
        category_id = ann["category_id"]
        x1, y1, w, h = ann["bbox"]
        scale_ratio = min(640 / height, 640 / width)
        scale_w = w * scale_ratio
        scale_h = h * scale_ratio
        area = scale_w * scale_h
        assert area > 0
        if area <= 32 * 32:
            small_objects += 1
            if category_id == 1:
                num_holothurian_s += 1
            elif category_id == 2:
                num_echinus_s += 1
            elif category_id == 3:
                num_scallop_s += 1
            elif category_id == 4:
                num_starfish_s += 1
        elif area <= 96 * 96:
            medium_objects += 1
            if category_id == 1:
                num_holothurian_m += 1
            elif category_id == 2:
                num_echinus_m += 1
            elif category_id == 3:
                num_scallop_m += 1
            elif category_id == 4:
                num_starfish_m += 1

        else:
            large_objects += 1
            if category_id == 1:
                num_holothurian_l += 1
            elif category_id == 2:
                num_echinus_l += 1
            elif category_id == 3:
                num_scallop_l += 1
            elif category_id == 4:
                num_starfish_l += 1


print("Num_processed_imgs = ", sum_img_ids)
print("Total objects: ", num_objects)
print("Small objects: %d %f" % (small_objects, small_objects / num_objects))
print(
    "     holothurian: %d %f" % (num_holothurian_s, num_holothurian_s / small_objects)
)
print("         echinus: %d %f" % (num_echinus_s, num_echinus_s / small_objects))
print("         scallop: %d %f" % (num_scallop_s, num_scallop_s / small_objects))
print("        starfish: %d %f" % (num_starfish_s, num_starfish_s / small_objects))
print("Medium objects: %d %f" % (medium_objects, medium_objects / num_objects))
print(
    "     holothurian: %d %f" % (num_holothurian_m, num_holothurian_m / medium_objects)
)
print("         echinus: %d %f" % (num_echinus_m, num_echinus_m / medium_objects))
print("         scallop: %d %f" % (num_scallop_m, num_scallop_m / medium_objects))
print("        starfish: %d %f" % (num_starfish_m, num_starfish_m / medium_objects))
print("Large objects: %d %f" % (large_objects, large_objects / num_objects))
print(
    "     holothurian: %d %f" % (num_holothurian_l, num_holothurian_l / large_objects)
)
print("         echinus: %d %f" % (num_echinus_l, num_echinus_l / large_objects))
print("         scallop: %d %f" % (num_scallop_l, num_scallop_l / large_objects))
print("        starfish: %d %f" % (num_starfish_l, num_starfish_l / large_objects))


# loading annotations into memory...
# Done (t=0.10s)
# creating index...
# index created!

# Total images:  6671
# Num_processed_imgs =  6671
# Total objects:  63998
# Small objects: 28105 0.439154
#      holothurian: 459 0.016332
#          echinus: 23321 0.829781
#          scallop: 356 0.012667
#         starfish: 3969 0.141220
# Medium objects: 34418 0.537798
#      holothurian: 5882 0.170899
#          echinus: 18923 0.549800
#          scallop: 1221 0.035476
#         starfish: 8392 0.243826
# Large objects: 1475 0.023048
#      holothurian: 467 0.316610
#          echinus: 711 0.482034
#          scallop: 130 0.088136
#         starfish: 167 0.113220

# 构造数据
labels = ["Small", "Medium", "Large"]
data_holothurian = [num_holothurian_s, num_holothurian_m, num_holothurian_l]
data_echinus = [num_echinus_s, num_echinus_m, num_echinus_l]
data_scallop = [num_scallop_s, num_scallop_m, num_scallop_l]
data_starfish = [num_starfish_s, num_starfish_m, num_starfish_l]

index = np.arange(len(labels))
width = 0.15

# plt.rcParams["font.family"] = "Times New Roman"
# plots
fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
bar_a = ax.bar(
    index - width * 3 /2,
    data_holothurian,
    width,
    label="holothurian",
    color="#130074",
    ec="black",
    lw=0.5,
)
bar_b = ax.bar(
    index - width / 2,
    data_echinus,
    width,
    label="echinus",
    color="#CB181B",
    ec="black",
    lw=0.5,
)
bar_c = ax.bar(
    index + width / 2,
    data_scallop,
    width,
    label="scallop",
    color="#008B45",
    ec="black",
    lw=0.5,
)
bar_d = ax.bar(
    index + width * 3 / 2,
    data_starfish,
    width,
    label="starfish",
    color="#FDBF6F",
    ec="black",
    lw=0.5,
)

#柱子上的数字显示
total_holothurian = num_holothurian_s + num_holothurian_m + num_holothurian_l
total_echinus = num_echinus_s + num_echinus_m + num_echinus_l
total_scallop = num_scallop_s + num_scallop_m + num_scallop_l
total_starfish = num_starfish_s + num_starfish_m + num_starfish_l

per_holothurian = [round(num_holothurian_s / total_holothurian, 2), round(num_holothurian_m / total_holothurian, 2), round(num_holothurian_l / total_holothurian, 2)]

print("per_holothurian = ", per_holothurian)

per_echinus = [round(num_echinus_s / total_echinus, 2), round(num_echinus_m / total_echinus, 2), round(num_echinus_l / total_echinus, 2)]

print("per_echinus = ", per_echinus)

per_scallop = [round(num_scallop_s / total_scallop, 2), round(num_scallop_m / total_scallop, 2), round(num_scallop_l / total_scallop, 2)]

print("per_scallop = ", per_scallop)

per_starfish = [round(num_starfish_s / total_starfish, 2), round(num_starfish_m / total_starfish, 2), round(num_starfish_l / total_starfish, 2)]

print("per_starfish = ", per_starfish)

for a, b, c in zip(index - width * 3 /2, data_holothurian, per_holothurian):
    ax.text(a, b, '%.2f'%c, ha="center", va="bottom", fontsize=6)

for a, b, c in zip(index - width / 2, data_echinus, per_echinus):
    ax.text(a, b, '%.2f'%c, ha="center", va="bottom", fontsize=6)

for a, b, c in zip(index + width / 2, data_scallop, per_scallop):
    ax.text(a, b, '%.2f'%c, ha="center", va="bottom", fontsize=6)

for a, b, c in zip(index + width * 3 / 2, data_starfish, per_starfish):
    ax.text(a, b, '%.2f'%c, ha="center", va="bottom", fontsize=6)

# 定制化设计
ax.tick_params(axis="x", direction="in", bottom=False)
ax.tick_params(axis="y", direction="out", labelsize=8, length=3)
ax.set_ylabel("Number of objects", fontsize = 'small')
ax.set_xticks(index)
ax.set_xticklabels(labels,fontsize = 'small')

for spine in ["top", "right"]:
    ax.spines[spine].set_color("none")

ax.legend(fontsize=7, frameon=False)

text_font = {"size": "14", "weight": "bold", "color": "black"}

plt.savefig(r"./statistics_duo.png", dpi=900, bbox_inches="tight")
plt.show()
