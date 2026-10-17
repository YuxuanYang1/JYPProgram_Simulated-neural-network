import numpy as np
import os
import csv

class XORNet:
    def __init__(self, layers, lr=0.1, seed=42):
        """
        layers: 每层神经元数，例如 [2, 4, 1]
        """
        np.random.seed(seed)
        self.layers = layers
        self.lr = lr

        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            std = np.sqrt(2.0 / layers[i])
            self.W.append(np.random.randn(layers[i], layers[i+1]) * std)
            self.b.append(np.zeros((1, layers[i+1])))

    # ---------- 激活 ----------
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # ---------- 前向 ----------
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        n = len(self.W)
        for i in range(n):
            z = a @ self.W[i] + self.b[i]
            self.z_values.append(z)
            if i == n - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            self.activations.append(a)
        return a

    # ---------- 反向 ----------
    def backward(self, X, y):
        m = X.shape[0]
        n = len(self.W)
        delta = self.activations[-1] - y
        for i in range(n - 1, -1, -1):
            dW = self.activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.z_values[i-1] > 0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    # ---------- 损失 ----------
    @staticmethod
    def compute_loss(y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ---------- 存档 ----------
    def save(self, path):
        arrays = {}
        for i, w in enumerate(self.W):
            arrays[f"W{i}"] = w
        for i, bb in enumerate(self.b):
            arrays[f"b{i}"] = bb
        arrays["layers"] = np.array(self.layers)
        np.savez(path, **arrays)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        n = (len(data.files) - 1) // 2
        self.W = [data[f"W{i}"] for i in range(n)]
        self.b = [data[f"b{i}"] for i in range(n)]
        self.layers = list(data["layers"])
        print(f"已加载模型: layers={self.layers}")

    # ---------- 预测 ----------
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


def train_xor(
    X, y,
    layers=(2, 4, 1),
    lr=0.5,
    epochs=5000,
    checkpoint_path="checkpoint.npz",
    state_path="state.npz",
    history_csv="history.csv",
    history_npz="history.npz",
    log_every=100,
    resume=True,
):
    """
    训练 XOR，支持断点续训 + 训练历史落盘。
    历史保存在:
      - history_csv: 每 log_every 轮写一行（可读、可用 Excel 打开）
      - history_npz: 每轮都追加（完整曲线，供 numpy 分析）
    """

    net = XORNet(layers=layers, lr=lr, seed=42)

    start_epoch = 0
    best_loss = float('inf')

    # 历史（完整）
    all_epochs = []
    all_losses = []

    # ---------- 恢复 ----------
    if resume and os.path.exists(checkpoint_path) and os.path.exists(state_path):
        print("发现存档，正在恢复...")
        net.load(checkpoint_path)
        state = np.load(state_path, allow_pickle=True)
        start_epoch = int(state['epoch'])
        best_loss = float(state['best_loss'])
        all_epochs = list(state['all_epochs'])
        all_losses = list(state['all_losses'])
        print(f"从 Epoch {start_epoch} 继续，历史已有 {len(all_epochs)} 条\n")
    else:
        print("从头开始训练\n")
        # 新建 CSV，写表头
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])

    # ---------- 训练 ----------
    try:
        for epoch in range(start_epoch, epochs):
            y_pred = net.forward(X)
            loss = net.compute_loss(y_pred, y)
            net.backward(X, y)

            # 记录每一轮
            all_epochs.append(epoch)
            all_losses.append(float(loss))

            if loss < best_loss:
                best_loss = loss

            # 定期打印 + 写 CSV
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:6d} | Loss: {loss:.6f} | Best: {best_loss:.6f}")
                with open(history_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f"{loss:.8f}"])

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，准备存档...")

    finally:
        # ---------- 存档 ----------
        print(f"\n正在存档 (epoch={start_epoch + len(all_epochs) - 1})...")
        net.save(checkpoint_path)
        np.savez(
            state_path,
            epoch=start_epoch + len(all_epochs),
            best_loss=best_loss,
            all_epochs=np.array(all_epochs),
            all_losses=np.array(all_losses),
        )
        # 完整历史也落一份
        np.savez(
            history_npz,
            epochs=np.array(all_epochs),
            losses=np.array(all_losses),
        )
        print("✅ 存档完成")
        print(f"   模型: {checkpoint_path}")
        print(f"   状态: {state_path}")
        print(f"   历史CSV: {history_csv}")
        print(f"   历史NPZ: {history_npz}")

    return net, all_epochs, all_losses


# ================== 主程序 ==================
if __name__ == "__main__":
    mode=input("训练模式？")
    if mode == "xor":
        # XOR 数据
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
    elif mode == "and":
        # AND 数据
        X = np.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]], dtype=float)
        y = np.array([[0],[0],[0],[1]], dtype=float)
    elif mode == "or":
        # OR 数据
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]], dtype=float)
        y = np.array([[0],[1],[1],[1]], dtype=float)

    net, epochs, losses = train_xor(
        X, y,
        layers=(2, 4, 1),
        lr=0.5,
        epochs=5000,
        log_every=100,
        resume=True,
    )

    print("\n最终预测：")
    pred = net.predict(X)
    prob = net.forward(X)
    for i in range(len(X)):
        print(f"  输入 {X[i].astype(int)} | 真实 {int(y[i][0])} | "
              f"预测 {pred[i][0]} | 概率 {prob[i][0]:.4f}")

    # ---------- 画训练曲线（如果装了 matplotlib） ----------
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, label="loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("XOR Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        print("\n已保存曲线: xor_loss_curve.png")
        plt.show()
    except ImportError as e:
        print(f"\n（matplotlib 导入失败: {e}）")
