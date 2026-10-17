import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 5))
for name in ["and", "or", "xor"]:
    data = np.load(f"{name}_history.npz")
    plt.plot(data["epochs"], data["losses"], label=name.upper())

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("AND / OR / XOR 的收敛速度")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("compare_speed.png")
plt.show()
