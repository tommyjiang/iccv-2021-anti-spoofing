import cv2
import numpy as np
import os
def generate_FT(root_pth, pic_pth, save_pth="./"):
    pic_name = pic_pth.split('/')[1]+".png"
    res_name = pic_pth.split('/')[1]+'_fft.png'
    print(os.path.join(root_pth,pic_pth))
    image = cv2.imread(os.path.join(root_pth,pic_pth))
    print(image.shape)
    cv2.imwrite(os.path.join(save_pth,pic_name),image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    cv2.imwrite(os.path.join(save_pth,res_name),fimg*255)
    # return fimg

if __name__=="__main__":
    root_pth = "../../raw_data/phase1"
    fake_face = "train/2_69_1_6_1_3/0001.png"
    true_face0 = "train/2_69_0_6_3_3/0009.png"
    true_face1 = "train/2_69_1_1_1_2/0002.png"
    generate_FT(root_pth,fake_face)
    generate_FT(root_pth,true_face0)
    generate_FT(root_pth,true_face1)


