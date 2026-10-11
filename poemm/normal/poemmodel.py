import numpy as np
import random
import os
from collections import Counter

class CharRNN:
    def __init__(self, vocab_size, hidden_size=32, lr=0.01, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        def rand_mat(r, c):
            std = np.sqrt(1.0 / r) * 0.1
            return np.random.randn(r, c) * std

        self.Wxh = rand_mat(vocab_size, hidden_size)
        self.Whh = rand_mat(hidden_size, hidden_size)
        self.Why = rand_mat(hidden_size, vocab_size)
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def forward(self, inputs, targets, h_prev):
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev.copy()
        loss = 0.0
        for t in range(len(inputs)):
            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0
            xs[t] = x
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ hs[t - 1] + self.bh)
            hs[t] = h
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
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0
            dby += dy
            dWhy += np.outer(hs[t], dy)
            dh = self.Why @ dy + dhnext
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += np.outer(xs[t], dh_raw)
            dWhh += np.outer(hs[t - 1], dh_raw)
            dhnext = self.Whh @ dh_raw

        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -1.0, 1.0, out=d)

        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def sample(self, seed_char, max_len=10, temperature=0.8):
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
            if ch == '<':
                break
            result += ch
        return result

    def save(self, path):
        np.savez(path,
                 Wxh=self.Wxh, Whh=self.Whh, Why=self.Why,
                 bh=self.bh, by=self.by)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.Wxh = data['Wxh']
        self.Whh = data['Whh']
        self.Why = data['Why']
        self.bh = data['bh']
        self.by = data['by']
        self.vocab_size = self.Wxh.shape[0]
        self.hidden_size = self.Wxh.shape[1]


# ================== 主程序 ==================
if __name__ == "__main__":
    with open("poems.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"原始: {len(lines)} 句")

    counter = Counter("".join(lines))
    common = {c for c, n in counter.items() if n >= 5}
    lines = ["".join(c if c in common else "<U>" for c in line) for line in lines]

    chars = sorted(set("".join(lines)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    print(f"字符表: {vocab_size}\n")

    rnn = CharRNN(vocab_size, hidden_size=32, lr=0.01, seed=42)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char

    checkpoint_path = "checkpoint.npz"
    state_path = "train_state.npz"

    start_epoch = 0
    best_loss = float('inf')
    wait = 0
    current_epoch = start_epoch

    if os.path.exists(checkpoint_path) and os.path.exists(state_path):
        print("发现存档，正在恢复...")
        rnn.load(checkpoint_path)
        rnn.char_to_idx = char_to_idx
        rnn.idx_to_char = idx_to_char
        state = np.load(state_path)
        start_epoch = int(state['epoch'])
        best_loss = float(state['best_loss'])
        wait = int(state['wait'])
        current_epoch = start_epoch
        print(f"从 Epoch {start_epoch} 继续训练\n")
    else:
        print("从头开始训练\n")

    epochs = 30000
    check_every = 1
    patience = 30
    min_improve = 1e-5
    target_loss = 0.5

    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            total_loss = 0.0
            if epoch % 10 == 0:
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

    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，准备存档...")

    finally:
        print(f"\n正在存档 (epoch={current_epoch})...")
        rnn.save(checkpoint_path)
        np.savez(state_path,
                 epoch=current_epoch + 1,
                 best_loss=best_loss,
                 wait=wait)
        print("✅ 存档完成，下次运行会从这里继续。")

    print(f"\n最终 Loss: {best_loss:.4f}")

    print("\n生成诗：\n")
    for sc in "床白春锄明黄夜花红孤千":
        if sc in char_to_idx:
            print(f"  {sc} → {rnn.sample(sc, max_len=10, temperature=0.8)}")
