import argparse
import pickle

import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview', 'campus', 'shelf'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha', default=1, help='weighted summation')
    parser.add_argument('--inter', default=1, type=int)
    arg = parser.parse_args()

    dataset = arg.datasets
    if arg.inter:
        label = open('./data/' + dataset + '/val_label.pkl', 'rb')
        r1 = open('./work_dir/' + dataset + '/agcn_test_joint/epoch11_test_score.pkl', 'rb')
        r2 = open('./work_dir/' + dataset + '/agcn_test_bone/epoch11_test_score.pkl', 'rb') 
    else:
        label = open('./data/' + dataset + '/val_label2.pkl', 'rb')
        r1 = open('./work_dir/' + dataset + '/agcn_test_joint2/epoch11_test_score.pkl', 'rb')
        r2 = open('./work_dir/' + dataset + '/agcn_test_bone2/epoch11_test_score.pkl', 'rb')
    
    label = np.array(pickle.load(label))
    r1 = list(pickle.load(r1).items())
    r2 = list(pickle.load(r2).items())
    print(len(r1), len(r2), len(label[0]))
    right_num = total_num = right_num_5 = 0
    #class_action = {"No_Interaction": 0, "Walk": 1, "Stand": 2, "HandRaise": 3, "Talk": 4, "Greet": 5, "HandRaiseStand": 6}
    class_id2 = ["No", "Walk", "Stand", "Put", "Handraise", "Carry", "Squat", "Bendover"]
    class_id1 = ["No_Interaction", "Talk", "Pass", "Fight"]
    if arg.inter:
        class_id = class_id1
    else:    
        class_id = class_id2
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        r = r11 + r22 * arg.alpha
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_ = np.argmax(r)
        right_num += int(r_ == int(l))
        total_num += 1
        #print("label: ", l ,r, r_)
        print("truth  : ", class_id[int(l)])
        print("predict: ", class_id[int(r_)])
        #print("truth: ", class_id1[int(l)%4], "  /  ", class_id2[int(l)//4])
        #print("predict: ", class_id1[int(r)%4], "  /  ", class_id2[int(r)//4])
        print()
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print("top1: {:.2f}%  right: {}  total: {}".format(acc*100, right_num, total_num))
    print("top5: {:.2f}%  right: {}  total: {}".format(acc5*100, right_num_5, total_num))
    #print(class_id)
    #print("top1: ", acc, "  top5: ", acc5)
