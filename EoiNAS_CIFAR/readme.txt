在搜索的过程中，每去掉一个候选operation，在上一轮搜索得到的网络内部（卷积核）参数和网络结构参数的基础上开始新一轮的搜索；修正网络结构参数的更新方式，该版本网络结构参数更新正确；加入gumbel-softmax采样；加入新指标；
dropout = 0.1；epoches=160;

CUDA_VISIBLE_DEVICES=0 python train_search.py
