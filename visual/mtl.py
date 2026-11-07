import os
import random
import numpy as np
from blom import XORNet


# 三个逻辑门的真值表
GATES = {
    "and": np.array([[0], [0], [0], [1]], dtype=float),
    "or":  np.array([[0], [1], [1], [1]], dtype=float),
    "xor": np.array([[0], [1], [1], [0]], dtype=float),
}

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)


def train_xor(
    gate="xor",
    hidden_size=4,
    lr=0.5,
    epochs=5000,
    log_every=100,
    callback=None,
    stop_flag=None,
    model_path=None,
    history_path=None,
):
    """
    训练逻辑门网络，支持断点续训。

    参数:
        gate: "and" / "or" / "xor"
        hidden_size: 隐藏层神经元数（若为 0 则不使用隐藏层，即 (2,1) 结构）
        lr: 学习率
        epochs: 总轮数
        log_every: 每多少轮回调一次
        callback: 函数 (epoch, loss)，loss 为 nan 表示“已恢复存档”
        stop_flag: 函数，返回 True 时停止训练
        model_path: 模型路径，默认 f"{gate}_model.npz"
        history_path: 历史路径，默认 f"{gate}_history.npz"

    返回:
        net, all_losses
    """
    gate = gate.lower()
    if gate not in GATES:
        raise ValueError(f"未知逻辑门: {gate}，可选 {list(GATES.keys())}")

    y = GATES[gate]
    model_path = model_path or f"{gate}_model.npz"
    history_path = history_path or f"{gate}_history.npz"

    # 结构：hidden_size > 0 用隐藏层，否则单层
    layers = [2, hidden_size, 1] if hidden_size > 0 else [2, 1]

    net = XORNet(layers=layers, lr=lr, seed=42)

    start_epoch = 0
    all_losses = []
    if os.path.exists(model_path) and os.path.exists(history_path):
        net.load(model_path)
        data = np.load(history_path, allow_pickle=True)
        all_losses = list(data["losses"])
        start_epoch = len(all_losses)
        if callback:
            callback(-1, float("nan"))

    for epoch in range(start_epoch, epochs):
        if stop_flag and stop_flag():
            break

        y_pred = net.forward(X)
        loss = net.compute_loss(y_pred, y)
        net.backward(X, y)
        all_losses.append(float(loss))

        if epoch % log_every == 0 and callback:
            callback(epoch, float(loss))

    net.save(model_path)
    np.savez(history_path,
             epochs=np.arange(len(all_losses)),
             losses=np.array(all_losses))

    return net, all_losses


def predict_gate(net):
    """返回 4 个输入的预测结果和概率"""
    prob = net.forward(X)
    pred = (prob > 0.5).astype(int)
    return pred, prob