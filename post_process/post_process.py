import cv2
import numpy as np
from tqdm import tqdm

det_labels_dict = {}
with open('./det_labels.txt', "r") as f:
    for line in f.readlines():
        name, det_label = line.strip().split(' ')
        idx = int(name.split('.')[0])
        det_label = int(det_label)
        det_labels_dict[idx] = det_label

pred = []
with open("./620b/pesudo_ckpt/pesudo_scores_multi_1.txt", "r") as f:
    for line in f.readlines():
        name, s0, s1, s2 = line.strip().split(' ')
        # name, s0 = line.strip().split(' ')
        
        idx = int(name.split('.')[0])
        score = float(s0)
        pred.append([idx, score])
pred.sort()



def judgeLight(im, th):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh, mask = cv2.threshold(gray, thresh=1, maxval=1, type=0)
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
    hist /= sum(hist)
    mean = 0
    for i in range(len(hist)):
        mean += hist[i] * i
    var = 0
    for i in range(len(hist)):
        var += hist[i] * ((i - mean)**2)
    std = np.sqrt(var)
    if std[0] < th:
        return True
    else:
        return False

def post_process(pred, det_labels_dict, scheme="scheme1"):
    root = "../raw_data/phase1/test"
    buckets = [[] for i in range(len(pred) // 10)]
    print(len(buckets))
    res = []
    for i in tqdm(range(len(buckets))):
        start_idx = i * 10 + 1
        dir_name = "{:0>4d}".format((start_idx // 1000))
        m_array = []
        for j in range(10):
            idx = start_idx + j
            # im = cv2.imread(root + '/' + dir_name + '/{:0>4d}.png'.format(idx))
            if det_labels_dict[idx] == 2:
                noise_label = 0
            else:

                im = cv2.imread(root + '/' + dir_name + '/{:0>4d}.png'.format(idx))
                if judgeLight(im, 10):
                    noise_label = 1
                else:
                    noise_label = 0
            m_array.append([pred[idx-1][0], pred[idx-1][1], det_labels_dict[idx], noise_label])

        m_array = np.array(m_array)

        noise_mask = (m_array[:, 3] == 1)

        norm_mask = (m_array[:, 2] == 2) & (~noise_mask)
        hard_mask = (m_array[:, 2] == 1) & (~noise_mask)
        noface_mask = (m_array[:, 2] == 0) | (noise_mask)

        label_fill = 0.0
        if np.sum(norm_mask) == len(m_array):
            label_fill = np.mean(m_array[:, 1][norm_mask])
        else:
            if np.sum(norm_mask) != 0:
                label_fill = np.mean(m_array[:, 1][norm_mask])
            else:
                if np.sum(hard_mask) != 0:
                    label_fill = np.mean(m_array[:, 1][hard_mask])
                else:
                    label_fill = 0.0
        m_array[:, 1] = label_fill
        

        for idx, score, det_label, isnoise in m_array:
            res.append([idx, score])

    return res



        


res = post_process(pred, det_labels_dict, scheme="scheme1")

with open("temp_res_test_post_620b_1e.txt", "w") as f:
    f.write('\n'.join("{:0>4d}.png {:.5f}".format(int(s1), s2) for s1, s2 in res))
