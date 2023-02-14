import sys

sys.path.extend(['../'])
from data_gen.rotation import *
from tqdm import tqdm


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # 选中一个sample
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):#在sample中选中一个person
            if person.sum() == 0:#对当前矩阵所有内容求和
                continue
            if person[0].sum() == 0:#如果这个person的第0帧对应的所有内容=0
                index = (person.sum(-1).sum(-1) != 0)#sum(-1):求每一行的和。如果最后一帧的最后一个关节存在，index=1，否则为0
                tmp = person[index].copy() #复制这一帧的内容
                person *= 0#清空
                person[:len(tmp)] = tmp #全部赋值为相同的内容
            for i_f, frame in enumerate(person):#选中一帧
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:#如果当前这个人对应的这个帧之后的内容都为0
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))#np.ceil:计算大于等于改值的最小整数。 使用有意义的数据循环填充
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]#concatenate拼接。
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()#选取一个sample下的一个person，它的第1个节点定义为main_body_center
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask #减去main_body_center的内容

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')#将第一个人的右肩(jpt 8)和左肩(jpt 4)之间的骨与x轴平行
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])#向量的叉积
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
