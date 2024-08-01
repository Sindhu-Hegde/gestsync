import torch
import numpy as np

def run_func_in_parts(func, vid_emb, aud_emb, part_len, dim, device):
    """
    Run given function in parts, spliting the inputs on dimension dim 
    This is used to save memory when inputs too large to compute on gpu 
    """
    dist_chunk = []
    for v_spl, a_spl in list(
            zip(vid_emb.split(part_len, dim=dim),
                aud_emb.split(part_len, dim=dim))):
        dist_chunk.append(func(v_spl.to(device), a_spl.to(device)))
    dist = torch.cat(dist_chunk, dim - 1)
    return dist

def logsoftmax_2d(logits):
    # Log softmax on last 2 dims because torch won't allow multiple dims
    orig_shape = logits.shape
    logprobs = torch.nn.LogSoftmax(dim=-1)(
        logits.reshape(list(logits.shape[:-2]) + [-1])).reshape(orig_shape)
    return logprobs
