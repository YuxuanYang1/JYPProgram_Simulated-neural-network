import math, random

class CharRNN:
    def __init__(self, vocab_size, hidden_size=32, lr=0.01, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        # 权重初始化（小尺度，防止一开始就爆炸）
        def rand_mat(r, c):
            std = math.sqrt(1.0 / r) * 0.1
            return [[random.gauss(0, std) for _ in range(c)] for _ in range(r)]

        self.Wxh = rand_mat(vocab_size, hidden_size)   # 输入 -> 隐状态
        self.Whh = rand_mat(hidden_size, hidden_size)  # 隐状态 -> 隐状态
        self.Why = rand_mat(hidden_size, vocab_size)   # 隐状态 -> 输出
        self.bh = [0.0] * hidden_size
        self.by = [0.0] * vocab_size

    @staticmethod
    def softmax(z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    def forward(self, inputs, targets, h_prev):
        """inputs/targets: 字符索引列表；h_prev: 初始隐状态"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev[:]
        loss = 0.0

        for t in range(len(inputs)):
            # one-hot 输入
            x = [0.0] * self.vocab_size
            x[inputs[t]] = 1.0
            xs[t] = x

            # h_t = tanh(Wxh·x + Whh·h_{t-1} + bh)
            h = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.bh[i]
                for j in range(self.vocab_size):
                    s += self.Wxh[j][i] * x[j]
                for j in range(self.hidden_size):
                    s += self.Whh[j][i] * hs[t - 1][j]
                h[i] = self.tanh(s)
            hs[t] = h

            # y_t = Why·h + by
            y = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                s = self.by[i]
                for j in range(self.hidden_size):
                    s += self.Why[j][i] * h[j]
                y[i] = s
            ys[t] = y

            p = self.softmax(y)
            ps[t] = p
            loss += -math.log(max(p[targets[t]], 1e-8))

        return loss, xs, hs, ys, ps

    def backward(self, inputs, targets, xs, hs, ps, h_prev):
        dWxh = [[0.0] * self.hidden_size for _ in range(self.vocab_size)]
        dWhh = [[0.0] * self.hidden_size for _ in range(self.hidden_size)]
        dWhy = [[0.0] * self.vocab_size for _ in range(self.hidden_size)]
        dbh = [0.0] * self.hidden_size
        dby = [0.0] * self.vocab_size

        dhnext = [0.0] * self.hidden_size

        for t in reversed(range(len(inputs))):
            # 输出层梯度（softmax + 交叉熵）
            dy = ps[t][:]
            dy[targets[t]] -= 1.0

            for i in range(self.vocab_size):
                dby[i] += dy[i]
                for j in range(self.hidden_size):
                    dWhy[j][i] += dy[i] * hs[t][j]

            # 回传到隐状态
            dh = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                s = 0.0
                for i in range(self.vocab_size):
                    s += self.Why[j][i] * dy[i]
                dh[j] = s + dhnext[j]

            # tanh 导数
            dh_raw = [(1 - hs[t][j] ** 2) * dh[j] for j in range(self.hidden_size)]

            for j in range(self.hidden_size):
                dbh[j] += dh_raw[j]
                for i in range(self.vocab_size):
                    dWxh[i][j] += dh_raw[j] * xs[t][i]
                for i in range(self.hidden_size):
                    dWhh[i][j] += dh_raw[j] * hs[t - 1][i]

            dhnext = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                s = 0.0
                for i in range(self.hidden_size):
                    s += self.Whh[j][i] * dh_raw[i]
                dhnext[j] = s

        # 梯度裁剪（收紧到 [-1, 1]）
        def clip(g):
            return max(-1.0, min(1.0, g))

        # 更新参数
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Wxh[i][j] -= self.lr * clip(dWxh[i][j])
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.Whh[i][j] -= self.lr * clip(dWhh[i][j])
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                self.Why[i][j] -= self.lr * clip(dWhy[i][j])
        for j in range(self.hidden_size):
            self.bh[j] -= self.lr * clip(dbh[j])
        for i in range(self.vocab_size):
            self.by[i] -= self.lr * clip(dby[i])

    def sample(self, seed_char, length=10):
        """从种子字符开始生成一串字符"""
        h = [0.0] * self.hidden_size
        idx = self.char_to_idx[seed_char]
        result = seed_char

        for _ in range(length):
            x = [0.0] * self.vocab_size
            x[idx] = 1.0

            h_new = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.bh[i]
                for j in range(self.vocab_size):
                    s += self.Wxh[j][i] * x[j]
                for j in range(self.hidden_size):
                    s += self.Whh[j][i] * h[j]
                h_new[i] = self.tanh(s)
            h = h_new

            y = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                s = self.by[i]
                for j in range(self.hidden_size):
                    s += self.Why[j][i] * h[j]
                y[i] = s
            temperature = 0.5
            y_scaled = [v / temperature for v in y]
            p = self.softmax(y_scaled)

            # 按概率采样
            r = random.random()
            cum = 0.0
            idx = self.vocab_size - 1
            for i, prob in enumerate(p):
                cum += prob
                if r < cum:
                    idx = i
                    break
            result += self.idx_to_char[idx]

        return result


# ================== 训练 ==================
if __name__ == "__main__":

    with open("names.txt", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"加载了 {len(names)} 个名字")
    
    chars = sorted(set("".join(names)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)

    print(f"字符表 ({vocab_size} 个): {''.join(chars)}\n")

    rnn = CharRNN(vocab_size, hidden_size=64, lr=0.01, seed=42)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char

    print("开始训练（学习英文名字的拼写规律）...")
    epochs = 5000
    for epoch in range(epochs):
        total_loss = 0.0
        for name in names:
            h_prev = [0.0] * rnn.hidden_size   # 每个名字重置隐状态
            inputs = [char_to_idx[c] for c in name[:-1]]
            targets = [char_to_idx[c] for c in name[1:]]
            loss, xs, hs, ys, ps = rnn.forward(inputs, targets, h_prev)
            rnn.backward(inputs, targets, xs, hs, ps, h_prev)
            total_loss += loss

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {total_loss / len(names):.4f}")

    print("\n生成新名字（这些名字数据里没有，是它自己编的）：")
    for seed_char in "abcdefxyz":
        generated = rnn.sample(seed_char, length=6)
        print(f"  以 '{seed_char}' 开头 → {generated}")
