import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}


def data_loader(file):
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            # 三元组的格式：头实体，尾实体，关系  ！！！！！！！
            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list


def distanceL2(h, r, t):
    # L2距离
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    # L1距离
    return np.sum(np.fabs(h + r - t))


class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1

        self.loss = 0

    def emb_initialize(self):
        """
        关系和实体的向量初始化
        :return:
        """
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            # 从 -6/根号k到6/根号k 随机采样100个数字
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            # 对随机采样的数据进行单位化  将每个关系的编号对应一个100维的向量  存入字典中
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            # 跟上述处理操作一样
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        # 调用该方法之后，每个关系不再仅仅是一个编号，而是对应的向量
        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        # batch的总数  400
        nbatches = 400
        # 每一个batch喂入的三元组数量 1207
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            # 循环1000次 epochs = 1000
            start = time.time()   # 记录本轮训练的开始时间
            self.loss = 0

            for k in range(nbatches):
                # 循环400次  每次随机从三元组列表中取出1207个三元组
                Sbatch = random.sample(self.triple_list, batch_size)
                Tbatch = []  # 三元组对(元组)的列表

                for triple in Sbatch:
                    # 随机一个真实三元组对应的负例三元组（替换头实体或者替换尾实体）
                    corrupted_triple = self.Corrupt(triple)
                    if (triple, corrupted_triple) not in Tbatch:
                        Tbatch.append((triple, corrupted_triple))
                self.update_embeddings(Tbatch)

            end = time.time()   # 记录本轮训练的结束时间
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            # 每20轮保存一次结果  实体的向量和关系的向量
            if epoch % 20 == 0:
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        # 训练完成后保存一次结果
        print("写入文件...")
        with codecs.open("entity_50dim_batch400", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("relation50dim_batch400", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")

    def Corrupt(self, triple):
        """
        构造负例三元组
        :param triple: 真实三元组
        :return: 负例三元组
        """
        corrupted_triple = copy.deepcopy(triple)   # 深拷贝
        # 随机初始化一个值
        seed = random.random()
        if seed > 0.5:
            # 替换head实体
            rand_head = triple[0]
            while rand_head == triple[0]:
                # 直到随机取样的实体和原三元组的头实体不同，结束循环
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0] = rand_head

        else:
            # 替换tail实体
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        """
        更新实体和关系的表示
        :param Tbatch: 三元组对  格式：[(true triple,false triple),(..,..),...]
        :return:
        """
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:

            # 更新的vector
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err

                # 关于L1范数的求导方法：参考了[刘知远组实现](https://github.com/thunlp/KB2E)中的实现
                # 先对L2范数求导，逐元素判断正负，为正赋值为1，负则为 - 1
                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # 梯度下降法---更新的关键公式
                # head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                # corrupt项整体为负，因此符号与correct相反
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                # relation更新两次
                relation_update -= self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg

        # 每一个batch之后做一个归一化
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)


if __name__ == '__main__':
    file1 = "FB15k/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=1001)
