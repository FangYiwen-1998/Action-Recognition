import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import csv

num_joint = 17  #  13
max_frame = 1
num_person_out = 1
num_person_in = 4

class_action = {"No": 0, "Walk": 1, "Stand": 2, "Handraise": 3, "Put": 4, "Carry": 5, "Squat": 6, "Bendover": 7}
class_inter = {"No_Interaction": 0, "Talk": 1, "Pass": 2, "Fight": 3}


class Feeder_campus(Dataset):
    """ 
    # Joint index:
    # {0,  "Nose"},
    # {1, "LEye"},
    # {2, "REye"},
    # {3, "LEar"},
    # {4, "REar"},
    # {5,  "LShoulder"},
    # {6,  "RShoulder"},
    # {7,  "LElbow"},
    # {8,  "RElbow"},
    # {9,  "LHand"},
    # {10,  "RHand"},
    # {11, "LHip"},
    # {12,  "RHip"},
    # {13, "LKnee"},
    # {14,  "RKnee"},
    # {15, "LAnkle"},
    # {16, "RAnkle"},
    
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=1,
                 num_person_out=1, 
                 inter=False):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample
        self.inter = inter
        self.label = []
        self.samples = []
        self.sample_id = []
        self.out_person = []
        
        self.load_datas()
        
    def load_datas(self):
        if not isinstance(self.data_path, list):
            self.data_path = [self.data_path]
        if not isinstance(self.label_path, list):
            self.label_path = [self.label_path]
        for i, data_path in enumerate(self.data_path):
            if self.inter:
                self.load_data(data_path, self.label_path[i])
            else:
                self.load_data2(data_path, self.label_path[i])
            
    def load_data2(self, data_path, label_path):
        tmps = []
        sample_seq = []
        k = 0
        with open(data_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                #if i == 0:
                #    continue
                sample_seq.append(k)
                k += 1
                tmp = np.zeros((num_person_out, 3, num_joint), dtype=np.float32)
                for j, r in enumerate(row):
                    if r:
                        rr = r.split(" ")
                        rrr = []
                        for a in rr:
                            a = a.strip(" ").strip("\n").strip("[").strip("]") 
                            if a:
                                rrr.append(a)
                        #rrr = [a for a in rr if a]
                        length = len(rrr)
                        xyz = [rrr[0:length//3], rrr[length//3:length//3*2], rrr[length//3*2:]]
                        tmp[0, :, :] = xyz
                        tmps.append(tmp)
                    else:
                        pass
                
                
        with open(label_path) as f:
            label_infos = json.load(f)
         
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person
        last_label = ""
        sample = np.zeros((self.C, self.T, self.V, self.M), dtype=np.float32)
        t, q = 0, len(self.sample_id)
        for img_id in sample_seq:
            label_info = label_infos[img_id]
            relation = label_info["relation"]
            number = label_info["Person_number"]
            #inter_label = class_inter[relation[0]["Interaction"]]
            current_label = ""
            for i, p in enumerate(relation):
                if i==0:
                    continue
                else:
                    self.sample_id.append(q)
                    q += 1
                    label = class_action[p["Interaction"]]
                    self.label.append(label)
                    for c in range(self.C):
                        sample[c, 0, :, 0] = tmps[t][0, c, :]
                    self.samples.append(sample)
                    sample = np.zeros((self.C, self.T, self.V, self.M), dtype=np.float32)
                    t += 1
                    
        print(len(self.samples), len(self.label))
        #print(self.label)
        
        
    def load_data(self, data_path, label_path):
        tmps = []
        sample_seq = []
        
        k = 0
        with open(data_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                #if i == 0:
                #    continue
                sample_seq.append(k)
                k += 1
                tmp = np.zeros((num_person_in, 3, num_joint), dtype=np.float32)
                for j, r in enumerate(row):
                    if r:
                        rr = r.split(" ")
                        rrr = []
                        for a in rr:
                            a = a.strip(" ").strip("\n").strip("[").strip("]") 
                            if a:
                                rrr.append(a)
                        #rrr = [a for a in rr if a]
                        length = len(rrr)
                        xyz = [rrr[0:length//3], rrr[length//3:length//3*2], rrr[length//3*2:]]
                        tmp[j, :, :] = xyz
                    else:
                        pass
                tmps.append(tmp)
                
        with open(label_path) as f:
            label_infos = json.load(f)
        
        #self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_in  # person
        last_label = ""
        sample = np.zeros((self.C, self.T, self.V, self.M), dtype=np.float32)
        t, q = 0, len(self.sample_id)
        number = 0
        for img_id in sample_seq:
            label_info = label_infos[img_id]
            relation = label_info["relation"]
            number = label_info["Person_number"]
            label = class_inter[relation[0]["Interaction"]]
            current_label = ""
            for i, p in enumerate(relation):
                if i==0:
                    continue
                else:
                    current_label += p["Interaction"]
            if current_label == last_label:
                pass
                #for c in range(self.C):
                #    for m in range(number):
                #        sample[c, t, :, m] = self.tmps[img_id][m, c, :]
                #t += 1
            else:
                if img_id != 0:
                    self.samples.append(sample)
                    self.out_person.append(number)
                sample = np.zeros((self.C, self.T, self.V, self.M), dtype=np.float32)
                t = 0
                self.sample_id.append(q)
                q += 1
                self.label.append(label)
            for c in range(self.C):
                for m in range(number):
                    sample[c, t, :, m] = tmps[img_id][m, c, :]
            #t += 1
            last_label = current_label
        self.samples.append(sample) 
        self.out_person.append(number)
        print(len(self.samples), len(self.label))
        print(self.label)
        
    def __getitem__(self, index):
        if self.inter:
            return self.__getitem2__(index)
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in), dtype=np.float32)
        sample = self.samples[index]
        label = self.label[index]
        #out_person = self.out_person[index]
        data_numpy = sample[:, :, :, :] # 为什么要复制呢
        #data_numpy = sample
        return data_numpy, label
        
        

    def load_data_(self):
        # load file list
        #self.sample_name = os.listdir(self.data_path)
        self.sample_id = []
        self.samples = []
        k = 0
        with open(self.data_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                #self.sample_id.append(i-1)
                sample = np.zeros((4, 3, 17), dtype=np.float32)
                for j , r in enumerate(row):
                    self.sample_id.append(k)
                    k += 1
                    if r:
                        #print(r)
                        rr = r.split(" ")
                        #print(rr)
                        rrr = [a for a in rr if a]
                        #print(rrr)
                        rrrr = ",".join(rrr)
                        #print(rrrr)
                        xyz = eval(rrrr)
                        
                        sample[j, :, :] = xyz
                    else:
                        pass
                    
                    self.samples.append(sample)
        #print(self.samples[0])
        # load label
        label_path = self.label_path
        print(label_path)
        with open(label_path) as f:
            label_infos = json.load(f)
        self.label = np.array([0 for i in self.sample_id])
        print(len(self.sample_id))
        for i in range(0, len(self.sample_id), 4):
            img_id = i // 4
            label_info = label_infos[img_id]
            relation = label_info["relation"]
            inter = 0
            inter_len = len(class_inter.keys())
            for p, re in enumerate(relation):
                if p == 0:
                    inter = class_inter[re["Interaction"]]
                else:
                    self.label[i+p-1] = class_action[re["Interaction"]]+inter_len*inter
            
        print(self.label)
        #self.label = np.array([label_info[_id]['label_index'] for _id in sample_id])
        #self.label = np.array([0 for i in self.sample_id])

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_id)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem2__(self, index):

        # output shape (C, T, V, M)
        # get data
        #sample_name = self.sample_name[index]
        #sample_path = os.path.join(self.data_path, sample_name)
        #with open(sample_path, 'r') as f:
        #    video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_out), dtype=np.float32)
        
        #for frame_info in video_info['data']:
        #    frame_index = frame_info['frame_index']
        #    for m, skeleton_info in enumerate(frame_info["skeleton"]):
        #        if m >= self.num_person_in:
        #            break
        #        pose = skeleton_info['pose']
        #        score = skeleton_info['score']
        #        data_numpy[0, frame_index, :, m] = pose[0::2]
        #        data_numpy[1, frame_index, :, m] = pose[1::2]
        #        data_numpy[2, frame_index, :, m] = score
        #for sample in self.samples:
        sample = self.samples[index]
        for j in range(3):
            data_numpy[j, 0, :, :] = sample[j][0]
                
       
        # centralization
        #data_numpy[0:2] = data_numpy[0:2] - 0.5
        #data_numpy[1:2] = -data_numpy[1:2]
        #data_numpy[0][data_numpy[2] == 0] = 0
        #data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        #label = video_info['label_index']
        #assert (self.label[index] == label)
        label = self.label[index]

        # sort by score
        #sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        #for t, s in enumerate(sort_index):
        #    data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label

def gendata(data_path, label_path,
            data_out_path, label_out_path, part,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=4,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_campus(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame,
        inter=True)

    sample_id = feeder.sample_id

    #8:2
    lens = len(sample_id)
    if part == 'train':
        sample_id = sample_id[:int(lens*0.8)]
    else:
        sample_id = sample_id[int(lens*0.8):]
        
    sample_label = []

    fp = np.zeros((len(sample_id), 3, max_frame, num_joint, num_person_in), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_id)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)
    print("sample_label: \n")
    print(sample_label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_id, list(sample_label)), f)

    np.save(data_out_path, fp)
    
    print("\nfinal: ", len(sample_id))

def gendata2(data_path, label_path,
            data_out_path, label_out_path, part,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_campus(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_id = feeder.sample_id
    lens = len(sample_id)
    if part == 'train':
        sample_id = sample_id[:int(lens*0.8)]
    else:
        sample_id = sample_id[int(lens*0.8):]
    sample_label = []

    fp = np.zeros((len(sample_id), 3, max_frame, num_joint, num_person_in), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_id)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)
    print("sample_label: \n")
    print(sample_label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_id, list(sample_label)), f)

    np.save(data_out_path, fp)
    
    print("\nfinal: ", len(sample_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Campus Data Converter.')
    parser.add_argument(
        '--data_path', default='./data/campus_raw')
    parser.add_argument(
        '--out_folder', default='./data/campus')
    parser.add_argument(
        '--inter', default=1, type=int)
    arg = parser.parse_args()
    
    part = ['val', "train"]
    for p in part:
        print('campus ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = []
        label_path = []
        for i in range(0, 1):
            dp = '{}/pose{}.csv'.format(arg.data_path, i)
            lp = '{}/train{}.json'.format(arg.data_path, i)
            data_path.append(dp)
            label_path.append(lp)
        if arg.inter:
            data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)
            label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

            gendata(data_path, label_path, data_out_path, label_out_path, p)
        else:
            data_out_path = '{}/{}_data_joint2.npy'.format(arg.out_folder, p)
            label_out_path = '{}/{}_label2.pkl'.format(arg.out_folder, p)

            gendata2(data_path, label_path, data_out_path, label_out_path, p)
