# 2s-AGCN
Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition in CVPR1
[2s-AGCN](https://www.jiqizhixin.com/articles/2020-01-03-4)

# NTU-RGB+D        https://arxiv.org/pdf/1604.02808.pdf
[NTU-RGB+D](https://blog.csdn.net/weixin_51450749/article/details/111768242)

# Skeleton-Kinetics    https://deepmind.com/research/open-source/kinetics
[Skeleton-Kinetics](https://blog.csdn.net/XCCCCZ/article/details/119836307?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4-119836307-blog-125510050.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4-119836307-blog-125510050.pc_relevant_default&utm_relevant_index=9)



# 执行步骤
 - 见README.md
 - 注意修改config里面的yaml配置
	`device,   GPU编号,   0或[0, ]或[0, 1, 2, 3]
	`batch_size,  遇到pickle问题或内存泄漏时,减小此值
	`test_batch_size,   同batch_size
	`weights,     模型参数,测试时需要加载的模型位置,注意名字尾部的数字
 - main.py里的参数
	`config
	`save-interval    大于1时模型没有保存一直在内存，有可能导致内存不足
	`eval-interval	  同save-interval
	`num-worker
	`weights		  从加载的模型参数开始训练
	`start-epoch	  配合weights使用
 - 出现pickle问题和内存泄漏
	`尝试减小batch_size、test_batch_size、save-interval、eval-interval、num-worker
	`测试数据太大，注释掉self.eval(),只执行self.train()
	


# 模型的使用
 - 使用main.py,  其中phase为test,  参数详情见get_parser()函数
 - 制作自己的test_joint.yaml/test_bone.yaml文件,使用main,py执行,
 - 自己的数据可参考data_gen里的ntu_gendata.py或kinetics-gendata.py转换,最后执行gen_bone_data.py
 - view和subject的唯一区别是数据集的划分方式
 - ensemble.py只是将测试结果统计比较,测试结构使用main.py加test_joint.yaml/test_bone.yaml将运行结构保存在work_dir中
 
 
# 交互动作和单人动作分开训练、分开识别 
 - 单人动作一每一帧的每一个人为输入数据，识别动作 
 - 交互动作以连续相同标签的多帧的多人为输入数据，识别动作
 
 
# 运行

 - csv和json数据放在data/campus_raw下，更改名为poseX.csv和trainX.json，X为数字


 - 生成关节数据joint和label
    `python data_gen/campus_gendata.py   --inter 1`  生成train_data_joint.npy、train_label.pkl和val_data_joint.npy、val_label.pkl
	`python data_gen/campus_gendata.py   --inter 0`  生成train_data_joint2.npy、train_label2.pkl和val_data_joint2.npy、val_label2.pkl
	
	
 - 生成骨骼数据bone
 
    `python data_gen/gen_campus_bone.py  --inter 1`  生成train_data_bone.npy和val_data_bone.npy
	`python data_gen/gen_campus_bone.py  --inter 0`  生成train_data_bone2.npy和val_data_bone2.npy
	
	
 - 训练
	`python main.py --config ./config/campus/train_joint.yaml`
	`python main.py --config ./config/campus/train_bone.yaml`
	运行以上两行，训练交互动作的关节和骨骼
	
	`python main.py --config ./config/campus/train_joint2.yaml`
	`python main.py --config ./config/campus/train_bone2.yaml`
	运行以上两行，训练单人动作的关节和骨骼	
	
	
 - 测试
	`python main.py --config ./config/campus/test_joint.yaml`
	`python main.py --config ./config/campus/test_bone.yaml`
	运行以上两行，测试交互动作，并将打分结果保持
    
	`python main.py --config ./config/campus/test_joint2.yaml`
	`python main.py --config ./config/campus/test_bone2.yaml`
	运行以上两行，测试单人动作，并将打分结果保持
	
	
 - 打分
	`python ensemble.py --datasets campus  --inter 1` 展示交互动作的打分结果
	`python ensemble.py --datasets campus  --inter 0` 展示单人动作的打分结果

	