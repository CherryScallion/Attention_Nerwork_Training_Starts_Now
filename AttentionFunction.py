#In this script, we will parse the attention data from a text file and convert it into input and output matrices.
#And then we will create a classical attention mechanism function 𝐴(𝑞,𝐾,𝑉) = Σ𝑝(𝑎(𝑘𝑖,𝑞)) × 𝑣𝑖
#And then we train this simple neural network using this attention mechanism.
import numpy as np
import torch
import re
import numpy as np
import os
import torch
import torch.nn as nn
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, "TrainingData.txt")
def parse_attention_data(data_string):
    vector_pattern = re.compile(r'\[(-?\d+),\s*(-?\d+)\]')# Regular expression to match vectors like [x, y](this is excusive for this specific dataset we created)
    matches = vector_pattern.findall(data_string)# Find all matches in the input string
    vectors = [list(map(float, match)) for match in matches]# Convert matched strings to float lists
    input_vectors = []
    output_vectors = []
    for i in range(0, len(vectors), 10):
        input_vectors.extend(vectors[i:i+5])
        output_vectors.extend(vectors[i+5:i+10])#A smart way to split the input and output vectors
    input_matrices = np.array(input_vectors).reshape(-1, 5, 2)
    output_matrices = np.array(output_vectors).reshape(-1, 5, 2)
    return input_matrices, output_matrices
with open(training_dir, 'r') as file:
    data = file.read()
input_data, output_data = parse_attention_data(data)

# print the shapes and first few matrices to verify
print("Input shapes:", input_data.shape)
print("The first input matrix:\n", input_data[0])
print("\nOut put matrix", output_data.shape)
print("The first output matrix:\n", output_data[0])
# Now, let's define the classical attention mechanism function 𝐴(𝑞,𝐾,𝑉) = Σ 𝑝(𝑎(𝑘𝑖,𝑞)) × 𝑣𝑖
# 
d_model = 2

# 定义 Query, Key, Value 的维度（d_k, d_v）
# 通常为了简化，它们与 d_model 相同
d_k = 2
d_v = 2

# 创建三个可训练的线性层，它们就是 Wq, Wk, Wv 矩阵
# nn.Linear(in_features, out_features, bias=True)
# in_features 是输入向量的维度
# out_features 是输出向量的维度
W_q = nn.Linear(d_model, d_k, bias=False)  # Query 矩阵
W_k = nn.Linear(d_model, d_k, bias=False)  # Key 矩阵
W_v = nn.Linear(d_model, d_v, bias=False)  # Value 矩阵

# 检查创建的矩阵（线性层的权重）
print("W_q 的权重矩阵形状:", W_q.weight.shape)
print("W_k 的权重矩阵形状:", W_k.weight.shape)
print("W_v 的权重矩阵形状:", W_v.weight.shape)

class Classic_Attention_Module:
    def __init__(self):
        pass
    def 
    def attention(self, q, K, V):
        """
        计算经典注意力机制
        q: 查询向量 (2,)
        K: 键矩阵 (5, 2)
        V: 值矩阵 (5, 2)
        """
        #  a(ki, q) = ki · q
        scores = torch.matmul(K, q)  # (5,)
        a = torch.dot(scores, q)
        A = torch.softmax(a, dim=0)
        # 
