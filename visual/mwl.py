import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal

from mtl import train_xor, predict_gate, GATES, X


class LogicThread(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, gate, hidden_size, lr, epochs, log_every):
        super().__init__()
        self.gate = gate
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.log_every = log_every
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            net, losses = train_xor(
                gate=self.gate,
                hidden_size=self.hidden_size,
                lr=self.lr,
                epochs=self.epochs,
                log_every=self.log_every,
                callback=lambda e, l: self.progress.emit(e, l),
                stop_flag=lambda: not self._running,
            )

            # 训练完，显示 4 个输入的预测
            pred, prob = predict_gate(net)
            y_true = GATES[self.gate].astype(int)

            lines = [f"训练完成！最终 loss: {losses[-1]:.4f}\n"]
            lines.append(f"逻辑门: {self.gate.upper()}\n")
            lines.append(f"{'输入':<12}{'真实':<8}{'预测':<8}{'概率':<10}")
            lines.append("-" * 40)
            for i in range(len(X)):
                inp = f"[{int(X[i][0])}, {int(X[i][1])}]"
                lines.append(
                    f"{inp:<12}{int(y_true[i][0]):<8}{pred[i][0]:<8}{prob[i][0]:.4f}"
                )

            self.finished.emit("\n".join(lines))
        except Exception as e:
            self.error.emit(str(e))


class LogicWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("逻辑门实验")
        self.setMinimumSize(560, 540)
        self.thread = None

        layout = QVBoxLayout()

        # 选择逻辑门
        gate_group = QGroupBox("逻辑门")
        gl = QHBoxLayout()
        gl.addWidget(QLabel("选择："))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["AND", "OR", "XOR"])
        self.gate_combo.setCurrentText("XOR")
        gl.addWidget(self.gate_combo)
        gate_group.setLayout(gl)
        layout.addWidget(gate_group)

        # 参数
        param_group = QGroupBox("训练参数")
        form = QFormLayout()

        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(0, 64)
        self.hidden_spin.setValue(4)
        self.hidden_spin.setSpecialValueText("无隐藏层")
        form.addRow("隐藏层神经元：", self.hidden_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 2.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.5)
        form.addRow("学习率：", self.lr_spin)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(100, 100000)
        self.epoch_spin.setValue(5000)
        self.epoch_spin.setSingleStep(500)
        form.addRow("训练轮数：", self.epoch_spin)

        self.log_spin = QSpinBox()
        self.log_spin.setRange(1, 1000)
        self.log_spin.setValue(100)
        form.addRow("日志频率：", self.log_spin)

        param_group.setLayout(form)
        layout.addWidget(param_group)

        # 按钮
        bl = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        layout.addLayout(bl)

        # 日志
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("训练日志...")
        layout.addWidget(self.log)

        self.setLayout(layout)

    def log_msg(self, m):
        self.log.append(m)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def start_training(self):
        gate = self.gate_combo.currentText().lower()
        hidden = self.hidden_spin.value()

        # 无隐藏层时，XOR 学不会，给个提醒
        if gate == "xor" and hidden == 0:
            self.log_msg("⚠ XOR 线性不可分，无隐藏层通常学不会，建议隐藏层 >= 2")

        self.log_msg(
            f"开始训练 {gate.upper()} | 隐藏层={hidden if hidden > 0 else '无'} | "
            f"lr={self.lr_spin.value()} | epochs={self.epoch_spin.value()}\n"
        )

        self.thread = LogicThread(
            gate,
            hidden,
            self.lr_spin.value(),
            self.epoch_spin.value(),
            self.log_spin.value(),
        )
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread.start()

    def stop_training(self):
        if self.thread:
            self.thread.stop()
            self.log_msg("⏹ 正在停止...")

    def on_progress(self, e, l):
        if np.isnan(l):
            self.log_msg(f"📦 已从 Epoch {e} 恢复")
        else:
            self.log_msg(f"Epoch {e:6d} | Loss: {l:.6f}")

    def on_finished(self, msg):
        self.log_msg(f"\n✅ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, msg):
        self.log_msg(f"\n❌ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)