# mte.py
import os
import random
import numpy as np
from ecnm import CharRNN


def train_rnn(
    lines,
    hidden_size=64,
    lr=0.005,
    epochs=5000,
    log_every=100,
    callback=None,
    stop_flag=None,
    model_path="model.npz",
    history_path="history.npz",
    max_lines=None,
):
    """
    训练字符 RNN，支持断点续训。

    参数:
        lines: 文本行列表（名字/诗句）
        hidden_size: 隐藏层神经元数
        lr: 学习率
        epochs: 总轮数
        log_every: 每多少轮回调一次
        callback: 函数 (epoch, loss)，loss 为 nan 表示“已恢复存档”
        stop_flag: 函数，返回 True 时停止训练（内层循环也会检查）
        model_path: 模型保存路径
        history_path: 历史保存路径
        max_lines: 若不为 None，只取前 max_lines 行（调试用）

    返回:
        rnn, all_losses
    """
    if max_lines is not None:
        lines = lines[:max_lines]

    chars = sorted(set("".join(lines)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)

    rnn = CharRNN(vocab_size, hidden_size=hidden_size, lr=lr)
    rnn.char_to_idx = char_to_idx
    rnn.idx_to_char = idx_to_char

    # ---------- 断点续训 ----------
    start_epoch = 0
    all_losses = []
    if os.path.exists(model_path) and os.path.exists(history_path):
        rnn.load(model_path)
        rnn.char_to_idx = char_to_idx
        rnn.idx_to_char = idx_to_char
        data = np.load(history_path, allow_pickle=True)
        all_losses = list(data["losses"])
        start_epoch = len(all_losses)
        if start_epoch > 0 and callback:
            callback(start_epoch, float("nan"))

    # ---------- 训练 ----------
    for epoch in range(start_epoch, epochs):
        if stop_flag and stop_flag():
            break

        total_loss = 0.0
        random.shuffle(lines)

        for name in lines:
            # 内层也检查停止，避免“正在停止”卡很久
            if stop_flag and stop_flag():
                break

            h_prev = np.zeros(rnn.hidden_size)
            inputs = [char_to_idx[c] for c in name[:-1]]
            targets = [char_to_idx[c] for c in name[1:]]
            loss, xs, hs, ps = rnn.forward(inputs, targets, h_prev)
            rnn.backward(inputs, targets, xs, hs, ps, h_prev)
            total_loss += loss

        avg_loss = total_loss / len(lines)
        all_losses.append(avg_loss)

        if epoch % log_every == 0 and callback:
            callback(epoch, float(avg_loss))

    # ---------- 保存 ----------
    rnn.save(model_path)
    np.savez(history_path,
             epochs=np.arange(len(all_losses)),
             losses=np.array(all_losses))

    return rnn, all_losses


def generate_samples(rnn, n=5, length=6, temperature=0.8):
    """生成 n 个样例"""
    seeds = list(rnn.char_to_idx.keys())
    return [
        rnn.sample(random.choice(seeds), length=length, temperature=temperature)
        for _ in range(n)
    ]