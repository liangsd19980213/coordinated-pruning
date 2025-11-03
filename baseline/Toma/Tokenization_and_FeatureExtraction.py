import javalang
import Levenshtein
import pandas as pd
import time
import multiprocessing as mp
import tqdm
import math
from functools import partial
import os
import logging

# ========== 修改部分 BEGIN ==========
# 替换输入路径：统一使用新的 CSV 文件路径
inputcsv = "bcb_test.csv"   # 可改为 bcb_valid.csv / bcb_test.csv
clone_type = os.path.splitext(os.path.basename(inputcsv))[0]  # bcb_train -> 作为输出文件名前缀
inputpath = os.path.join(os.path.dirname(__file__), "id2sourcecode")  # Java 源码路径
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
# ========== 修改部分 END ============

def get_sim(tool, dataframe):
    sim = []
    for _, pair in dataframe.iterrows():
        id1, id2 = pair.FuncID1, pair.FuncID2
        sourcefile1 = os.path.join(inputpath, f"{id1}.java")
        sourcefile2 = os.path.join(inputpath, f"{id2}.java")
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log = f"{time.asctime()} {tool} {id1} {id2} {similarity}"
            logging.error(log)
            similarity = 'False'
        # print(similarity)
        sim.append(similarity)

    return sim

def getCodeBlock_type(file_path):
    block = []
    if not os.path.exists(file_path):
        return block
    with open(file_path, 'r') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                token_type = str(type(token))[:-2].split(".")[-1]
                block.append(token_type)
    return block

def runner(tool, sourcefile1, sourcefile2):
    block1 = getCodeBlock_type(sourcefile1)
    block2 = getCodeBlock_type(sourcefile2)
    if tool == 't1':
        return Jaccard_sim(block1, block2)
    elif tool == 't2':
        return Dice_sim(block1, block2)
    elif tool == 't3':
        return Jaro_sim(block1, block2)
    elif tool == 't4':
        return Jaro_winkler_sim(block1, block2)
    elif tool == 't5':
        return Levenshtein_sim(block1, block2)
    elif tool == 't6':
        return Levenshtein_ratio(block1, block2)

def intersection_and_union(group1, group2):
    intersection = 0
    union = 0
    triads_num1 = {}
    triads_num2 = {}
    for triad1 in group1:
        triads_num1[triad1] = triads_num1.get(triad1, 0) + 1
    for triad2 in group2:
        triads_num2[triad2] = triads_num2.get(triad2, 0) + 1

    for triad in list(set(group1).union(set(group2))):
        intersection += min(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
        union += max(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
    return intersection, union

def Jaccard_sim(group1, group2):
    intersection, union = intersection_and_union(group1, group2)
    sim = float(intersection) / union if union != 0 else 0
    return sim

def Dice_sim(group1, group2):
    intersection, union = intersection_and_union(group1, group2)
    sim = 2 * float(intersection) / (len(group1) + len(group2)) if (len(group1) + len(group2)) != 0 else 0
    return sim

def Jaro_sim(group1, group2):
    return Levenshtein.jaro(group1, group2)

def Jaro_winkler_sim(group1, group2):
    return Levenshtein.jaro_winkler(group1, group2)

def Levenshtein_sim(group1, group2):
    return Levenshtein.distance(group1, group2)

def Levenshtein_ratio(group1, group2):
    return Levenshtein.ratio(group1, group2)

def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num / n))
    df_split = []
    for index in range(n):
        if index < n - 1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split

def main(pairs, clone_type):
    df_split = cut_df(pairs, 60)

    for method, tool in zip(['t1', 't2', 't3', 't4', 't5', 't6'], range(1, 7)):
        func = partial(get_sim, f't{tool}')
        pool = mp.Pool(processes=4)
        sim_results = []
        it = tqdm.tqdm(pool.imap(func, df_split), total=len(df_split))
        for item in it:
            sim_results.extend(item)
        pool.close()
        pool.join()
        pairs[f't{tool}_sim'] = sim_results

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    pairs.to_csv(os.path.join(output_dir, f"{clone_type}_sim.csv"), index=False)


if __name__ == '__main__':
    logging.basicConfig(filename='errorlog.txt', level=logging.ERROR)
    start = time.time()

    inputcsv = "bcb_test.csv"
    clone_type = os.path.splitext(os.path.basename(inputcsv))[0]

    pairs = pd.read_csv(inputcsv)
    assert 'FuncID1' in pairs.columns and 'FuncID2' in pairs.columns, "Missing FuncID1/FuncID2 columns!"

    main(pairs, clone_type)

    print(f"Total time: {time.time() - start:.2f} seconds")

