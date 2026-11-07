import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal

from mte import train_rnn, generate_samples


class PoemsThread(QThread):
    progress = pyqtSignal(int, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, data_file, hidden, lr, epochs, log_every):
        super().__init__()
        self.data_file = data_file
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.log_every = log_every
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            with open(self.data_file, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            rnn, losses = train_rnn(
                lines,
                hidden_size=self.hidden,
                lr=self.lr,
                epochs=self.epochs,
                log_every=self.log_every,
                model_path="poems_model.npz",
                history_path="poems_history.npz",
                callback=lambda e, l: self.progress.emit(e, l),
                stop_flag=lambda: not self._running,
            )

            # 生成样例：用几个常见字做种子
            samples = []
            for seed in "床白春锄明黄夜花红孤千":
                if seed in rnn.char_to_idx:
                    samples.append(f"{seed} → {rnn.sample(seed, length=10, temperature=0.8)}")

            self.finished.emit(
                f"训练完成！最终 loss: {losses[-1]:.4f}\n\n"
                + "\n".join(samples)
            )
        except Exception as e:
            self.error.emit(str(e))


class PoemsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("古诗生成")
        self.setMinimumSize(560, 520)
        self.thread = None

        layout = QVBoxLayout()

        # 数据文件
        file_group = QGroupBox("数据文件")
        fl = QHBoxLayout()
        fl.addWidget(QLabel("poems.txt："))
        self.file_input = QLineEdit("poems.txt")
        fl.addWidget(self.file_input)
        file_group.setLayout(fl)
        layout.addWidget(file_group)

        # 参数
        param_group = QGroupBox("训练参数")
        form = QFormLayout()
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(16, 256)
        self.hidden_spin.setValue(64)
        form.addRow("隐藏层神经元：", self.hidden_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.005)
        form.addRow("学习率：", self.lr_spin)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(100, 100000)
        self.epoch_spin.setValue(5000)
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
        if not os.path.exists(self.file_input.text()):
            self.log_msg("❌ 文件不存在")
            return

        self.log_msg("开始训练...\n")
        self.thread = PoemsThread(
            self.file_input.text(),
            self.hidden_spin.value(),
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
            self.log_msg(f"Epoch {e:6d} | Loss: {l:.4f}")

    def on_finished(self, msg):
        self.log_msg(f"\n✅ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, msg):
        self.log_msg(f"\n❌ {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)