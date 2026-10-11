import math
import random

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.5, seed=42):
        """
        纯 Python 神经网络
        :param layers: 每层神经元数量，例如 [2, 4, 1]
        """
        self.layers = layers
        self.learning_rate = learning_rate
        random.seed(seed)
        
        # 权重：weights[i] 是第 i 层的权重矩阵 (layers[i] x layers[i+1])
        # 偏置：biases[i] 是第 i 层的偏置向量（长度为 layers[i+1]）
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            # He 初始化
            std = math.sqrt(2.0 / layers[i])
            w = [[random.gauss(0, std) for _ in range(layers[i + 1])]
                 for _ in range(layers[i])]
            b = [0.0] * layers[i + 1]
            self.weights.append(w)
            self.biases.append(b)
    
    # ---------- 激活函数 ----------
    @staticmethod
    def sigmoid(x):
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def relu(x):
        return x if x > 0 else 0.0
    
    # ---------- 矩阵运算（纯 Python） ----------
    @staticmethod
    def matmul_vec(mat, vec):
        """矩阵 (m x n) 乘 向量 (n) -> 向量 (m)"""
        return [sum(mat[i][j] * vec[j] for j in range(len(vec)))
                for i in range(len(mat))]
    
    # ---------- 前向传播 ----------
    def forward(self, x):
        """
        x: 单个样本，形如 [0.0, 1.0]
        返回：输出向量
        """
        self.activations = [x]   # 每层的输出
        self.z_values = []       # 每层的加权输入
        
        a = x
        num_layers = len(self.weights)
        for i in range(num_layers):
            w, b = self.weights[i], self.biases[i]
            # z = W @ a + b
            z = [sum(w[j][k] * a[j] for j in range(len(a))) + b[k]
                 for k in range(len(b))]
            self.z_values.append(z)
            
            if i == num_layers - 1:
                a = [self.sigmoid(v) for v in z]     # 输出层
            else:
                a = [self.relu(v) for v in z]        # 隐藏层
            
            self.activations.append(a)
        
        return a
    
    # ---------- 反向传播 ----------
    def backward(self, x, y):
        """
        x: 单个输入样本
        y: 单个标签（标量 0/1 或 one-hot 列表）
        """
        num_layers = len(self.weights)
        # 输出层误差（二元交叉熵 + sigmoid）
        output = self.activations[-1]
        delta = [output[k] - y[k] for k in range(len(output))]
        
        for i in range(num_layers - 1, -1, -1):
            a_prev = self.activations[i]
            w = self.weights[i]
            b = self.biases[i]
            
            # 梯度计算
            dw = [[delta[k] * a_prev[j] for k in range(len(delta))]
                  for j in range(len(a_prev))]
            db = delta[:]
            
            # 传播到前一层（除了第一层之后不再往前）
            if i > 0:
                z_prev = self.z_values[i - 1]
                new_delta = []
                for j in range(len(w)):
                    # 误差回传：sum_k w[j][k] * delta[k]
                    s = sum(w[j][k] * delta[k] for k in range(len(delta)))
                    # ReLU 导数
                    new_delta.append(s if z_prev[j] > 0 else 0.0)
                delta = new_delta
            
            # 更新权重和偏置
            for j in range(len(w)):
                for k in range(len(w[j])):
                    w[j][k] -= self.learning_rate * dw[j][k]
            for k in range(len(b)):
                b[k] -= self.learning_rate * db[k]
    
    # ---------- 损失 ----------
    @staticmethod
    def compute_loss(y_pred, y_true):
        eps = 1e-8
        loss = 0.0
        for p, t in zip(y_pred, y_true):
            p = max(eps, min(1 - eps, p))
            loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        return loss / len(y_pred)
    
    # ---------- 训练 ----------
    def train(self, X, Y, epochs=5000, verbose=True):
        """X: 样本列表；Y: 标签列表（每个样本一个列表）"""
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in zip(X, Y):
                y_pred = self.forward(x)
                total_loss += self.compute_loss(y_pred, y)
                self.backward(x, y)
            avg_loss = total_loss / len(X)
            
            if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f}")
    
    def predict(self, x):
        out = self.forward(x)
        return 1 if out[0] > 0.5 else 0


# ================== 测试 XOR ==================
if __name__ == "__main__":
    X = [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]
    Y = [[0.0], [1.0], [1.0], [0.0]]
    
    nn = NeuralNetwork(layers=[2, 8, 1], learning_rate=0.5, seed=42)
    
    print("开始训练 XOR 问题（纯 Python 实现）：")
    nn.train(X, Y, epochs=50000, verbose=True)
    
    print("\n预测结果：")
    for x, y in zip(X, Y):
        prob = nn.forward(x)[0]
        pred = nn.predict(x)
        print(f"输入: {x}  真实: {int(y[0])}  预测: {pred}  概率: {prob:.4f}")
