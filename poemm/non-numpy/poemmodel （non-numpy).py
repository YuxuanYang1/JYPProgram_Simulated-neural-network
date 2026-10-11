import math, random

class CharRNN:
    def __init__(self, vocab_size, hidden_size=64, lr=0.01, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        def rand_mat(r, c):
            std = math.sqrt(1.0 / r) * 0.1
            return [[random.gauss(0, std) for _ in range(c)] for _ in range(r)]

        self.Wxh = rand_mat(vocab_size, hidden_size)
        self.Whh = rand_mat(hidden_size, hidden_size)
        self.Why = rand_mat(hidden_size, vocab_size)
        self.bh = [0.0] * hidden_size
        self.by = [0.0] * vocab_size

    @staticmethod
    def softmax(z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def forward(self, inputs, targets, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev[:]
        loss = 0.0

        for t in range(len(inputs)):
            x = [0.0] * self.vocab_size
            x[inputs[t]] = 1.0
            xs[t] = x

            h = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.bh[i]
                for j in range(self.vocab_size):
                    s += self.Wxh[j][i] * x[j]
                for j in range(self.hidden_size):
                    s += self.Whh[j][i] * hs[t - 1][j]
                h[i] = math.tanh(s)
            hs[t] = h

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
            dy = ps[t][:]
            dy[targets[t]] -= 1.0

            for i in range(self.vocab_size):
                dby[i] += dy[i]
                for j in range(self.hidden_size):
                    dWhy[j][i] += dy[i] * hs[t][j]

            dh = [0.0] * self.hidden_size
            for j in range(self.hidden_size):
                s = 0.0
                for i in range(self.vocab_size):
                    s += self.Why[j][i] * dy[i]
                dh[j] = s + dhnext[j]

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

        def clip(g):
            return max(-1.0, min(1.0, g))

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

    def sample(self, seed_char, max_len=10, temperature=0.8):
        """生成一句诗，遇到 <E> 就停"""
        h = [0.0] * self.hidden_size
        if seed_char not in self.char_to_idx:
            seed_char = random.choice(list(self.char_to_idx.keys()))
        idx = self.char_to_idx[seed_char]
        result = seed_char

        for _ in range(max_len):
            x = [0.0] * self.vocab_size
            x[idx] = 1.0

            h_new = [0.0] * self.hidden_size
            for i in range(self.hidden_size):
                s = self.bh[i]
                for j in range(self.vocab_size):
                    s += self.Wxh[j][i] * x[j]
                for j in range(self.hidden_size):
                    s += self.Whh[j][i] * h[j]
                h_new[i] = math.tanh(s)
            h = h_new

            y = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                s = self.by[i]
                for j in range(self.hidden_size):
                    s += self.Why[j][i] * h[j]
                y[i] = s

            y = [v / temperature for v in y]
            p = self.softmax(y)

            r = random.random()
            cum = 0.0
            idx = self.vocab_size - 1
            for i, prob in enumerate(p):
                cum += prob
                if r < cum:
                    idx = i
                    break

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
    epochs = 30000
    check_every = 50
    patience = 30
    min_improve = 1e-3
    target_loss = 0.5
    best_loss = float('inf')
    wait = 0

    print("开始训练（中文古诗 + 句末标记）...\n")

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(lines)
        for line in lines:
            h_prev = [0.0] * rnn.hidden_size
            inputs = [char_to_idx[c] for c in line[:-1]]
            targets = [char_to_idx[c] for c in line[1:]]
            loss, xs, hs, ys, ps = rnn.forward(inputs, targets, h_prev)
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
