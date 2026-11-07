import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from mwe import NamesWindow
from mwp import PoemsWindow
from mwl import LogicWindow


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经网络研究项目")
        self.setMinimumSize(420, 360)

        # 子窗口引用（防止被垃圾回收）
        self.child_windows = {}

        layout = QVBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("神经网络实验平台")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("选择一个实验进入")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)

        # 三个入口按钮
        btn_names = QPushButton("英文名字生成")
        btn_names.setMinimumHeight(48)
        btn_names.clicked.connect(self.open_names)
        layout.addWidget(btn_names)

        btn_poems = QPushButton("古诗生成")
        btn_poems.setMinimumHeight(48)
        btn_poems.clicked.connect(self.open_poems)
        layout.addWidget(btn_poems)

        btn_logic = QPushButton("逻辑门实验")
        btn_logic.setMinimumHeight(48)
        btn_logic.clicked.connect(self.open_logic)
        layout.addWidget(btn_logic)

        layout.addStretch()
        self.setLayout(layout)

    def _open(self, key, window_class):
        # 如果已经开过，直接激活；否则新建
        if key in self.child_windows and self.child_windows[key].isVisible():
            self.child_windows[key].activateWindow()
            self.child_windows[key].raise_()
            return
        win = window_class()
        self.child_windows[key] = win
        win.show()

    def open_names(self):
        self._open("names", NamesWindow)

    def open_poems(self):
        self._open("poems", PoemsWindow)

    def open_logic(self):
        self._open("logic", LogicWindow)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec_())