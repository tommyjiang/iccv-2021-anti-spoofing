import cv2
import math
import numpy as np
import random


def add_light(img,p=0.6,strength=250):
    """
    img: cv2格式的图像矩阵
    p: 添加光源的概率
    light_num: 添加光源的个数
    strength: 光源的强度
    """
    if random.uniform(0,1) >= p:
        return img
    rows,cols = img.shape[:2]
    center_x = random.randint(0,rows)
    center_y = random.randint(0,cols)
    min_radius = int(math.sqrt(math.pow(center_x,2)+math.pow(center_y,2)))
    max_radius = int(math.sqrt(math.pow(rows,2)+math.pow(cols,2)))
    radius = random.randint(min_radius,max_radius)
    dst = np.zeros((rows,cols,3), dtype = 'uint8')
    for i in range(rows):
        for j in range(cols):
            ## 计算当前点到光照中心的距离
            distance = math.pow((center_y-j),2) + math.pow((center_x-i),2)
            B = img[i,j][0]
            G = img[i,j][1]
            R = img[i,j][2]
            if distance<radius*radius:
                # 按照距离远近计算增强的光照
                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                dst[i,j] = np.uint8((B, G, R))
            else:
                dst[i,j] = np.uint8((B, G, R))
    return dst
if __name__ == '__main__':
    img = cv2.imread('./0002.png')
    for i in range(10):
        tar_img = add_light(img)
        cv2.imwrite('./after_light/%i.png'%i,tar_img)



# img = cv2.imread('./0002.png')

# rows,cols = img.shape[:2]
# center_x = 75
# center_y = 75
# radius = math.sqrt(math.pow(rows/2,2)+math.pow(cols/2,2))

# ## 添加光照强度
# strength = 200

# ## 新建目标输出图像

# dst = np.zeros((rows,cols,3), dtype = "uint8")
# for i in range(rows):
#     for j in range(cols):
#         ## 计算当前点到光照中心的距离
#         distance = math.pow((center_y-j),2) + math.pow((center_x-i),2)
#         B = img[i,j][0]
#         G = img[i,j][1]
#         R = img[i,j][2]
#         if distance<radius*radius:
#             # 按照距离远近计算增强的光照
#             result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
#             B = img[i,j][0] + result
#             G = img[i,j][1] + result
#             R = img[i,j][2] + result
#             B = min(255, max(0, B))
#             G = min(255, max(0, G))
#             R = min(255, max(0, R))
#             dst[i,j] = np.uint8((B, G, R))
#         else:
#             dst[i,j] = np.uint8((B, G, R))
# cv2.imwrite('./after_light.png',dst)


