import numpy as np
import random
from collections import Counter


class CharRNN:
    def __init__(self, vocab_size, hidden_size=64, lr=0.01, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        # He 初始化（小尺度，防爆炸）
        def rand_mat(r, c):
            std = np.sqrt(1.0 / r) * 0.1
            return np.random.randn(r, c) * std

        self.Wxh = rand_mat(vocab_size, hidden_size)   # 输入 -> 隐状态
        self.Whh = rand_mat(hidden_size, hidden_size)  # 隐状态 -> 隐状态
        self.Why = rand_mat(hidden_size, vocab_size)   # 隐状态 -> 输出
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def forward(self, inputs, targets, h_prev):
        """inputs/targets: 索引列表；h_prev: 初始隐状态"""
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0

        for t in range(len(inputs)):
            # one-hot 输入
            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0
            xs[t] = x

            # h_t = tanh(Wxh·x + Whh·h_{t-1} + bh)
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ hs[t - 1] + self.bh)
            hs[t] = h

            # y_t = Why·h + by
            y = self.Why.T @ h + self.by
            p = self.softmax(y)
            ps[t] = p
            loss += -np.log(max(p[targets[t]], 1e-8))

        return loss, xs, hs, ps

    def backward(self, inputs, targets, xs, hs, ps, h_prev):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dhnext = np.zeros(self.hidden_size)

        for t in reversed(range(len(inputs))):
            # 输出层梯度（softmax + 交叉熵）
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0

            dby += dy
            dWhy += np.outer(hs[t], dy)

            # 回传到隐状态
            dh = self.Why @ dy + dhnext

            # tanh 导数
            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw
            dWxh += np.outer(xs[t], dh_raw)
            dWhh += np.outer(hs[t - 1], dh_raw)

            dhnext = self.Whh @ dh_raw

        # 梯度裁剪
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -1.0, 1.0, out=d)

        # 更新参数
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def sample(self, seed_char, max_len=10, temperature=0.8):
        """生成一句，遇到 <E> 就停"""
        h = np.zeros(self.hidden_size)
        if seed_char not in self.char_to_idx:
            seed_char = random.choice(list(self.char_to_idx.keys()))
        idx = self.char_to_idx[seed_char]
        result = seed_char

        for _ in range(max_len):
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0

            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ h + self.bh)
            y = self.Why.T @ h + self.by

            y = y / temperature
            p = self.softmax(y)

            idx = np.random.choice(self.vocab_size, p=p)
            ch = self.idx_to_char[idx]
            if ch == '<':   # 遇到 <E> 的 < 就停
                break
            result += ch

        return result


# ================== 训练 ==================
if __name__ == "__main__":
    with open("poems.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"加载了 {len(lines)} 句诗")

    counter = Counter("".join(lines))
    common = {c for c, n in counter.items() if n >= 5}
    lines = ["".join(c if c in common else "<U>" for c in line) for line in lines]
    print(f"砍完后字符表: {len(set(''.join(lines)))}")
    
    all_text = "\n".join(lines)
    chars = sorted(set(all_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)

    print(f"字符表大小: {vocab_size}\n")

    rnn = CharRNN(vocab_size, hidden_size=64, lr=0.01, seed=42)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char

    # 训练参数
    epochs = 50000
    check_every = 1
    patience = 2000
    min_improve = 1e-3
    target_loss = 0.5
    best_loss = float('inf')
    wait = 0

    print("开始训练（NumPy 版，速度快几十倍）...\n")

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(lines)
        for line in lines:
            h_prev = np.zeros(rnn.hidden_size)
            inputs = [char_to_idx[c] for c in line[:-1]]
            targets = [char_to_idx[c] for c in line[1:]]
            loss, xs, hs, ps = rnn.forward(inputs, targets, h_prev)
            rnn.backward(inputs, targets, xs, hs, ps, h_prev)
            total_loss += loss

        avg_loss = total_loss / len(lines)

        if epoch % check_every == 0:
            improvement = best_loss - avg_loss
            if best_loss == float('inf'):
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | (基准)")
            else:
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | 进步: {improvement:+.5f}")

            if avg_loss < target_loss:
                print(f"\n达到目标 loss {target_loss}，停止。")
                break
            if improvement < min_improve:
                wait += 1
                if wait >= patience:
                    print(f"\n连续 {patience} 次无进步，停止。")
                    break
            else:
                 wait = 0
            best_loss = min(best_loss, avg_loss)

    print(f"\n训练结束，最终 Loss: {best_loss:.4f}")

    # ================== 生成诗 ==================
    print("\n让它写诗（给种子字，遇到句末自动停）：\n")
    seed_chars = "床白春锄明黄夜花红孤千"
    for sc in seed_chars:
        if sc in char_to_idx:
            poem = rnn.sample(sc, max_len=10, temperature=0.8)
            print(f"  {sc} → {poem}")
