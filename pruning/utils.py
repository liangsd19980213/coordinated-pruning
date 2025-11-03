import os, random, torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import json
import logging
from pruning.metric_logger import MetricLogger

logger = logging.getLogger(__name__)
metric_logger = MetricLogger()
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def nanstd(o, dim):
    return torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim)
    )


# def repack_tensor_and_create_mask(tensor, mask, fuse=False):
#     """
#     Given a `mask`, this function removes from `tensor` the tokens according to that mask and returns
#     the new batch tensor and updated mask.
#     If `fuse` is True, it will merge the masked tokens into one tensor which will be included in the new sequence.
#     """
#     batch = []
#     lengths = []
#     for el, msk in zip(tensor, mask):
#         new_len = msk.sum().item()
#         if fuse:
#             new_len += 1
#         _, hidden_dim = el.shape
#         _m = msk[..., None].bool()
#         if fuse:
#             new_el = el.masked_select(_m)
#             inv_m = ~_m
#             num_masked_tokens = inv_m.int().sum().item()
#             fused_tokens = el.masked_select(inv_m).reshape((num_masked_tokens, hidden_dim)).mean(0)
#             new_el = torch.cat((new_el, fused_tokens)).reshape((new_len, hidden_dim))
#         else:
#             new_el = el.masked_select(_m).reshape((new_len, hidden_dim))
#         batch.append(new_el)
#         lengths.append(new_len)
#
#     padded_batch = pad_sequence(batch, batch_first=True)
#     new_mask = (padded_batch > 0).any(-1)
#
#     return padded_batch, new_mask

def repack_tensor_and_create_mask(tensor, mask, fuse=False):
    """
    Safe version: Automatically skip invalid samples instead of crashing.
    """
    batch = []
    lengths = []

    for idx, (el, msk) in enumerate(zip(tensor, mask)):
        try:  # ====【新增】try-except 防止崩溃====
            seq_len, hidden_dim = el.shape
            if msk.shape[0] != seq_len:
                # ====【新增】非法mask长度时跳过并警告====
                print(f"[Warning] Skipping sample {idx}: Mask length {msk.shape[0]} != tensor length {seq_len}")
                continue

            new_len = msk.sum().item()
            _m = msk[..., None].bool()

            # ====【改动】处理mask全0的情况====
            if new_len == 0:
                if fuse:
                    fused_tokens = el.mean(0, keepdim=True)
                    new_el = fused_tokens
                    new_len = 1
                else:
                    new_el = torch.zeros((1, hidden_dim), device=el.device)
                    new_len = 1
            else:
                if fuse:
                    new_el = el.masked_select(_m)
                    inv_m = ~_m
                    num_masked_tokens = inv_m.int().sum().item()
                    if num_masked_tokens > 0:
                        fused_tokens = el.masked_select(inv_m).reshape((num_masked_tokens, hidden_dim)).mean(0)
                    else:
                        # ====【新增】避免空反mask导致mean()崩溃====
                        fused_tokens = torch.zeros((hidden_dim,), device=el.device)
                    new_el = torch.cat((new_el, fused_tokens)).reshape((new_len + 1, hidden_dim))
                    new_len += 1
                else:
                    selected = el.masked_select(_m)
                    if selected.numel() != new_len * hidden_dim:
                        # ====【新增】选中元素数量异常时跳过====
                        print(f"[Warning] Skipping sample {idx}: selected.numel() {selected.numel()} != {new_len * hidden_dim}")
                        continue
                    new_el = selected.reshape((new_len, hidden_dim))

            batch.append(new_el)
            lengths.append(new_len)

        except Exception as e:
            # ====【新增】捕获未知错误并跳过====
            print(f"[Warning] Skipping sample {idx} due to error: {e}")
            continue

    # ====【新增】避免整个batch都被跳过导致pad_sequence崩溃====
    if len(batch) == 0:
        dummy = torch.zeros((1, 1, tensor.shape[-1]), device=tensor.device)
        return dummy, torch.zeros((1, 1), dtype=torch.bool, device=tensor.device)

    padded_batch = pad_sequence(batch, batch_first=True)
    new_mask = (padded_batch > 0).any(-1)

    return padded_batch, new_mask


def save_dict_as_json(data, file_path):
    """
    Saves a given dictionary as a JSON file.

    Parameters:
    - data (dict): The dictionary to save as JSON.
    - file_path (str): The file path where the JSON file will be saved.

    Returns:
    - None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Dictionary saved successfully as JSON in {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary as JSON: {e}")

def compare_kl_divergence(curr_scores, prev_scores, layer_idx):
    """
    比较 KL 散度并根据内部设定的阈值判断是否跳过层。
    """
    kl_threshold = 0.00001  # ✅ 内部固定阈值，可统一修改

    kl_div = torch.nn.functional.kl_div(
        curr_scores.log_softmax(dim=-1),
        prev_scores.softmax(dim=-1),
        reduction='batchmean'
    ).item()

    # print(f"[KL] Layer {layer_idx}: KL divergence = {kl_div:.6f} (threshold = {kl_threshold:.6f})")

    if kl_div < kl_threshold:
        # logger.info(f"[SKIP] Layer {layer_idx} skipped due to low KL divergence.")
        return True, kl_div
    else:
        # logger.info(f"[FORWARD] Layer {layer_idx} executed.")
        return False, kl_div
def pad_scores(scores, target_len=512):
    """
    scores: Tensor of shape [bs, seq_len]
    returns: Tensor of shape [bs, target_len]
    """
    curr_len = scores.size(1)
    if curr_len < target_len:
        pad_size = target_len - curr_len
        padding = torch.zeros(scores.size(0), pad_size, device=scores.device, dtype=scores.dtype)
        scores = torch.cat([scores, padding], dim=1)
    else:
        scores = scores[:, :target_len]
    return scores





