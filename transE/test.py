import numpy as np
import codecs
import operator
import json
from transE import data_loader, entity2id, relation2id


def dataloader(entity_file, relation_file, test_file):
    """
    数据处理
    :param entity_file: 存储训练好的实体向量的文件
    :param relation_file: 存储训练好的关系向量的文件
    :param test_file: 存储测试三元组的文件
    :return:
    """
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            entity_dict[entity] = embedding

    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding

    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')
            if len(triple) != 3:
                continue
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            test_triple.append(tuple((h_, t_, r_)))

    return entity_dict, relation_dict, test_triple


def distance(h, r, t):
    """
    求L2距离
    :param h:
    :param r:
    :param t:
    :return:
    """
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h + r - t
    return np.linalg.norm(s)


class Test:
    def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=True):
        """
        :param entity_dict:
        :param relation_dict:
        :param test_triple:
        :param train_triple:
        :param isFit: 是否过滤掉出现在训练集中的三元组   当破坏头实体或者尾实体的时候，可能破坏后的三元组已经出现在训练集当中
        """
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    def rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            # triple格式：(h, t, r)
            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():
                # 破坏头实体得到的负例三元组  只要出现在训练集的实体中，都要替换，进行计算评分
                corrupted_head = [entity, triple[1], triple[2]]
                if self.isFit:
                    # 判断破坏的三元组不在训练三元组中
                    if corrupted_head not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_head[0]]
                        r_emb = self.relation_dict[corrupted_head[2]]
                        t_emb = self.entity_dict[corrupted_head[1]]
                        # 为该破坏的三元组进行打分
                        rank_head_dict[tuple(corrupted_head)] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_head[0]]
                    r_emb = self.relation_dict[corrupted_head[2]]
                    t_emb = self.entity_dict[corrupted_head[1]]
                    rank_head_dict[tuple(corrupted_head)] = distance(h_emb, r_emb, t_emb)
                # 破坏尾实体得到的负例三元组
                corrupted_tail = [triple[0], entity, triple[2]]
                if self.isFit:
                    if corrupted_tail not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_tail[0]]
                        r_emb = self.relation_dict[corrupted_tail[2]]
                        t_emb = self.entity_dict[corrupted_tail[1]]
                        # 为该破坏的三元组进行打分
                        rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_tail[0]]
                    r_emb = self.relation_dict[corrupted_tail[2]]
                    t_emb = self.entity_dict[corrupted_tail[1]]
                    rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)

            # items返回的是字典对应的元组的列表
            # operator.itemgetter 获取对象的哪些维的数据  这是一个函数
            # 根据元组索引为1的元素进行排序  从小到大
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

            # 计算评价指标  hits@10和MR
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    # 如果三元组排在前十名，hits加1，因为我计算的是hits@10
                    if i < 10:
                        hits += 1
                    # i表示的排名的索引，所以计算排名的时候应该加1
                    rank_sum = rank_sum + i + 1
                    break

            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            step += 1   # 用于控制打印结果的计数
            if step % 5000 == 0:
                print("step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
                print()
        # 有破坏头实体的，还有破坏尾实体的，所以应该除以2
        self.hits10 = hits / (2 * len(self.test_triple))
        self.mean_rank = rank_sum / (2 * len(self.test_triple))

    def relation_rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            rank_dict = {}
            for r in self.relation_dict.keys():
                corrupted_relation = (triple[0], triple[1], r)
                if self.isFit and corrupted_relation in self.train_triple:
                    continue
                h_emb = self.entity_dict[corrupted_relation[0]]
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                rank_dict[r] = distance(h_emb, r_emb, t_emb)

            # rand_sorted格式：[(2,0.001),(1,0.01),...]
            rank_sorted = sorted(rank_dict.items(), key=operator.itemgetter(1))

            rank = 1
            for i in rank_sorted:
                # i格式：(r,distance)
                # 如果输入的三元组是 (1,3,2)
                if triple[2] == i[0]:
                    break
                rank += 1
            if rank < 10:
                hits += 1
            rank_sum = rank_sum + rank + 1

            step += 1
            if step % 5000 == 0:
                print("relation step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
                print()

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)


if __name__ == '__main__':
    # 前面两个为占位符
    _, _, train_triple = data_loader("FB15k/")
    # 加载实体向量、关系向量和测试三元组
    entity_dict, relation_dict, test_triple = \
        dataloader("entity_50dim_batch400", "relation50dim_batch400",
                   "FB15k/test.txt")
    # 测试
    test = Test(entity_dict, relation_dict, test_triple, train_triple, isFit=False)
    test.rank()
    print("entity hits@10: ", test.hits10)
    print("entity meanrank: ", test.mean_rank)

    test.relation_rank()
    print("relation hits@10: ", test.relation_hits10)
    print("relation meanrank: ", test.relation_mean_rank)

    f = open("result.txt", 'w')
    f.write("entity hits@10: " + str(test.hits10) + '\n')
    f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    f.close()
