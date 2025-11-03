# import json
# import os
#
# jsonl_path = '../data/bcb/data.jsonl'
# output_dir = '../baseline/Toma/id2sourcecode'
#
# os.makedirs(output_dir, exist_ok=True)
#
# with open(jsonl_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         obj = json.loads(line)
#         idx = obj['idx']
#         code = obj['func']
#         with open(os.path.join(output_dir, f"{idx}.java"), 'w', encoding='utf-8') as fout:
#             fout.write(code)


import pandas as pd
import os

# 设置路径
base_dir = "../../data/bcb"
output_dir = ''
files = {
    "train.txt": "bcb_train.csv",
    "valid.txt": "bcb_valid.csv",
    "test.txt": "bcb_test.csv"
}

# 遍历生成三个 CSV 文件
for txt_file, csv_file in files.items():
    txt_path = os.path.join(base_dir, txt_file)
    csv_path = os.path.join(output_dir, csv_file)

    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                func1, func2, label = parts
                data.append([func1, func2, label])

    df = pd.DataFrame(data, columns=["FuncID1", "FuncID2", "Label"])
    df.to_csv(csv_path, index=False)
    print(f"✅ 已生成：{csv_path}")
