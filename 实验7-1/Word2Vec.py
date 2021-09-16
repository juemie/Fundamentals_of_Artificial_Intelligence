import collections
import random
import re

import jieba
import numpy as np

stop_worlds = []
with open("..data/stop_words.txt", 'r', encoding='utf-8') as f_stop_words:
    for line in f_stop_words:
        # 去掉停用词的中的回车，换行符，空格
        line = line.replace('\r', '').replace('\n', '').strip()
        # print(line)
        stop_worlds.append(line)
    # 打印停用词的长度
print(len(stop_worlds))
# 去掉重复后停用词的长度
stop_worlds = set(stop_worlds)
print(len(stop_worlds))

raw_word_list = []  # 分完词储存列表
rules = u"(\u4300-\u9fa5)"
pattern = re.compile(rules)
f_write = open("..data/Seg_The_Smiling_Proud_Wanderer.txt", "w", encoding="utf-8")
# 读取数据集
with open("data/The_Smiling_Proud_Wanderer.txt", "r", encoding="utf-8") as f_reader:
    lines = f_reader.readlines()
    for line in lines:
        # 去掉文本中的回车，换行符，空格
        line = line.replace('\r', '').replace('\n', '').strip()
        if line == '' or line is None:
            continue
        # jieba分词处理
        line = "".join(jieba.cut(line))
        seg_list = pattern.findall(line)
        # 去掉停用词的列表
        world_list = []
        for word in seg_list:
            if word not in world_list:
                world_list.append(word)
        if len(world_list) > 0:
            raw_word_list.extend(world_list)
            line = "".join(world_list)
            # line = "".join(seg_list)
            f_write.write(line + "\n")
            f_write.flush()
f_write.close()
print(raw_word_list)
print(len(raw_word_list))
print(set(raw_word_list))
# 文本编码通过汉字找到对应的编码，再通过编码找到对应的汉字
vocabulary_size = len(set(raw_word_list))
words = raw_word_list
# count存放每个词在文本出现的次数
count = [['UNK', '-1']]
count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
print("count", len(count))
dictionary = dict()

# 词的整形编码
for word, _ in count:
    dictionary[word] = len(dictionary)
data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0
        unk_count = unk_count + 1
    data.append(index)
count[0][1] = unk_count
# 根据编码找到对应的词
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
del words
print(reverse_dictionary[1000])
print(data[:200])

data_index = 0
"""
    批量数据之前，在训练之前需要一批一批送入数据
"""


def generate_batch(batch_size, num_ships, skip_window):
    global data_index
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    # 对某个单词创建相关样本时使用到的单词数量
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_ships):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_ships):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_ships + j] = buffer[skip_window]
            batch[i * num_ships + j] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=128, num_ships=4, skip_window=2)
for i in range(10):
    print(batch[i], reverse_dictionary[batch[i]], '-->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# skip-gram model
batch_size = 128  # batch_size为batch大小
embedding_size = 300
skip_window = 2
num_skips = 64
valid_window = 100
learning_rate = 0.01
