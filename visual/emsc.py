import numpy as np
import random
from collections import Counter

# ================== 加载模型 ==================
class CharRNN:
    def __init__(self, vocab_size, hidden_size=64):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @staticmethod
    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def forward_loss(self, inputs, targets, h_prev):
        """只做前向，返回平均交叉熵（用于 BPC 计算）"""
        hs = {}
        hs[-1] = h_prev.copy()
        loss = 0.0
        for t in range(len(inputs)):
            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ hs[t - 1] + self.bh)
            hs[t] = h
            y = self.Why.T @ h + self.by
            p = self.softmax(y)
            loss += -np.log(max(p[targets[t]], 1e-8))
        return loss / len(inputs)

    def sample(self, seed_char, length=6, temperature=0.8):
        h = np.zeros(self.hidden_size)
        if seed_char not in self.char_to_idx:
            seed_char = random.choice(list(self.char_to_idx.keys()))
        idx = self.char_to_idx[seed_char]
        result = seed_char
        for _ in range(length):
            x = np.zeros(self.vocab_size)
            x[idx] = 1.0
            h = np.tanh(self.Wxh.T @ x + self.Whh.T @ h + self.bh)
            y = self.Why.T @ h + self.by
            y = y / temperature
            p = self.softmax(y)
            idx = np.random.choice(self.vocab_size, p=p)
            result += self.idx_to_char[idx]
        return result

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.Wxh = data["Wxh"]
        self.Whh = data["Whh"]
        self.Why = data["Why"]
        self.bh = data["bh"]
        self.by = data["by"]
        self.vocab_size = self.Wxh.shape[0]
        self.hidden_size = self.Wxh.shape[1]


# ================== 读数据 ==================
with open("names.txt", encoding="utf-8") as f:
    all_names = [line.strip().lower() for line in f if line.strip()]
all_names = [n for n in all_names if n.isalpha()]

# 划分训练集 / 测试集（80% / 20%）
random.seed(42)
random.shuffle(all_names)
split = int(len(all_names) * 0.8)
train_names = all_names[:split]
test_names = all_names[split:]

print(f"总名字: {len(all_names)}，训练: {len(train_names)}，测试: {len(test_names)}")

chars = sorted(set("".join(all_names)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

# 加载模型
rnn = CharRNN(vocab_size, hidden_size=64)
rnn.load("names_model.npz")
rnn.char_to_idx = char_to_idx
rnn.idx_to_char = idx_to_char
print(f"模型加载完成: vocab={rnn.vocab_size}, hidden={rnn.hidden_size}\n")


# ================== 1. 算 BPC ==================
def compute_bpc(model, names):
    total_loss = 0.0
    total_chars = 0
    for name in names:
        if len(name) < 2:
            continue
        h_prev = np.zeros(model.hidden_size)
        inputs = [char_to_idx[c] for c in name[:-1]]
        targets = [char_to_idx[c] for c in name[1:]]
        loss = model.forward_loss(inputs, targets, h_prev)
        total_loss += loss * len(inputs)
        total_chars += len(inputs)
    return total_loss / total_chars / np.log(2)   # 转成 bits-per-character


train_bpc = compute_bpc(rnn, train_names)
test_bpc = compute_bpc(rnn, test_names)

print("=" * 50)
print("【1】BPC（bits-per-character）")
print(f"  训练集 BPC: {train_bpc:.4f}")
print(f"  测试集 BPC: {test_bpc:.4f}")
print(f"  差值: {test_bpc - train_bpc:+.4f}")
if test_bpc - train_bpc > 0.3:
    print("  ⚠ 测试集明显高于训练集，可能过拟合")
else:
    print("  ✅ 泛化良好")


# ================== 2. 规则命中率 ==================
vowels = set("aeiou")
illegal_clusters = ["xk", "qz", "xz", "qk", "zx", "kq"]

def rule_hit_rate(model, n=100, length=6, temperature=0.8):
    """生成 n 个名字，统计规则命中率"""
    names = []
    for _ in range(n):
        seed = random.choice(list(char_to_idx.keys()))
        if seed in vowels or seed.isalpha():
            generated = model.sample(seed, length=length, temperature=temperature)
            names.append(generated)

    total = len(names)
    if total == 0:
        return {}

    first_ok = sum(1 for x in names if x[0] == x[0])   # 首字母永远一致，占位
    len_ok = sum(1 for x in names if 3 <= len(x) <= 12)
    vowel_ok = sum(1 for x in names if any(c in vowels for c in x))
    illegal_ok = sum(1 for x in names if not any(cl in x for cl in illegal_clusters))

    return {
        "总数": total,
        "长度合规率(3~12)": len_ok / total,
        "元音存在率": vowel_ok / total,
        "无非法辅音簇率": illegal_ok / total,
    }


print("\n" + "=" * 50)
print("【2】规则命中率（100 个生成名字）")
rates = rule_hit_rate(rnn, n=100)
for k, v in rates.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2%}")
    else:
        print(f"  {k}: {v}")


# ================== 3. 原创性 ==================
def edit_distance_1(a, b):
    """判断两个字符串编辑距离是否为 1"""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return diff == 1
    # 长度差 1 的情况
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    diff = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diff += 1
            j += 1
            if diff > 1:
                return False
        else:
            i += 1
            j += 1
    return True


def originality_rate(model, train_set, n=100, length=6, temperature=0.8):
    generated = []
    for _ in range(n):
        seed = random.choice(list(char_to_idx.keys()))
        generated.append(model.sample(seed, length=length, temperature=temperature))

    exact = sum(1 for x in generated if x in train_set)
    variant = sum(1 for x in generated if x not in train_set and
                  any(edit_distance_1(x, t) for t in train_set))

    return {
        "生成数": n,
        "完全记忆率": exact / n,
        "变体率(编辑距离1)": variant / n,
        "原创率(既非记忆也非变体)": (n - exact - variant) / n,
    }


print("\n" + "=" * 50)
print("【3】原创性分析")
train_set = set(train_names)
orig = originality_rate(rnn, train_set, n=100)
for k, v in orig.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2%}")
    else:
        print(f"  {k}: {v}")


# ================== 4. 生成样例 ==================
print("\n" + "=" * 50)
print("【4】生成样例（10 个）")
for _ in range(10):
    seed = random.choice(list(char_to_idx.keys()))
    print(f"  {seed} → {rnn.sample(seed, length=6, temperature=0.8)}")
