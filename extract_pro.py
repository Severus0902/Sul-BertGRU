import random
import os

random.seed(42)
# 读取两个文件，分别存储为列表变量
with open("data/positive1.txt", 'r') as f:
    positive_sequences = f.readlines()[1::2]  # 获取偶数行，即蛋白质序列 从索引为 1 的元素开始，以步长为 2 获取整个序列

with open("data/negative1.txt", 'r') as f:
    negative_sequences = f.readlines()[1::2]  # 获取偶数行，即蛋白质序列

# 确定正样本的数量，将负样本序列列表随机打乱，并取出与正样本数量相同负样本序列
negative_count = len(negative_sequences)
positive_count = len(positive_sequences)
random.shuffle(negative_sequences)
negative_sequences = negative_sequences[:positive_count]

if not os.path.exists('neg'):
    os.makedirs('neg')
if not os.path.exists('pos'):
    os.makedirs('pos')

# 遍历正样本序列列表和负样本序列列表，对于每一个序列，生成一个seq文件，并将其存储到对应的正样本文件夹或负样本文件夹中
for i, sequence in enumerate(negative_sequences):
    with open(f"neg/{i + 1}.seq", "w") as f:
        f.write(f"{' '.join(sequence.strip())}")
        #f.write(f"{sequence.rstrip()}")

for i, sequence in enumerate(positive_sequences):
    with open(f"pos/{i + 1}.seq", "w") as f:
        f.write(f"{' '.join(sequence.strip())}")
        #f.write(f"{sequence.rstrip()}")
