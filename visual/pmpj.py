import json
import re

# 1. 读 JSON
with open("唐诗三百首.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"读入 {len(data)} 首诗")

lines = []

for poem in data:
    # 有些 JSON 结构里诗句在 paragraphs，有些在 content
    paras = poem.get("paragraphs") or poem.get("content") or []
    
    for para in paras:
        # 去掉标点，按句切分
        # 中文标点：，。！？；、
        parts = re.split(r"[，。！？；、]", para)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # 只要 5 或 7 个字的句子（五言/七言）
            if len(p) in (5, 7):
                lines.append(p + "<E>")

# 去重
lines = list(set(lines))
print(f"清洗后得到 {len(lines)} 句")

# 写回 poems.txt
with open("poems.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("已写入 poems.txt")
