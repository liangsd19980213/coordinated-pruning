import os
import csv
import json
from polp.utils.io import read_json, read_file

def save_csv(data, file_path):
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['func', 'label'])
        writer.writeheader()
        for item in data:
            writer.writerow({
                'func': item['function_body'],
                'label': item['label']
            })

def main():
    # 路径设置
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/devign'))

    json_file = os.path.join(root, 'functions.jsonl')
    train_ids_file = os.path.join(root, 'train.txt')
    valid_ids_file = os.path.join(root, 'valid.txt')
    test_ids_file = os.path.join(root, 'test.txt')

    # Step 1: 加载 functions.jsonl
    data = read_json(json_file)
    transformed_data = [{'function_body': item['func'], 'label': item['target']} for item in data]

    # Step 2: 加载 ID 文件
    def load_ids(path):
        return [int(line.strip()) for line in read_file(path)]

    train_ids = load_ids(train_ids_file)
    valid_ids = load_ids(valid_ids_file)
    test_ids  = load_ids(test_ids_file)

    # Step 3: 按 ID 匹配数据
    train_data = [transformed_data[i] for i in train_ids]
    valid_data = [transformed_data[i] for i in valid_ids]
    test_data  = [transformed_data[i] for i in test_ids]

    # Step 4: 保存 CSV
    save_csv(train_data, os.path.join(root, 'devign_train.csv'))
    save_csv(valid_data, os.path.join(root, 'devign_valid.csv'))
    save_csv(test_data,  os.path.join(root, 'devign_test.csv'))

    print(f"[✓] Saved {len(train_data)} train samples to devign_train.csv")
    print(f"[✓] Saved {len(valid_data)} valid samples to devign_valid.csv")
    print(f"[✓] Saved {len(test_data)} test  samples to devign_test.csv")

if __name__ == "__main__":
    main()
