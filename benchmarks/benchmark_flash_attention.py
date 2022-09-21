from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from benchmarks.utils import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

import xlwt

DEVICE_TFLOPS = 312

def attention_ref(qkv, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)


torch.manual_seed(0)
repeats = 30

dropout_p = 0.0
causal = False
dtype = torch.float16
device = 'cuda'

batch_sizes = [1, 2, 8, 32, 64]
num_heads = [16, 12]
seqlens = [1, 2, 16, 32, 128, 256]
head_dims = [64]
configs = []
for batch_size in batch_sizes:
    for num_head in num_heads:
        for seqlen in seqlens:
            for head_dim in head_dims:
                configs.append([batch_size, num_head, seqlen, head_dim])

workbook = xlwt.Workbook(encoding= 'ascii')
worksheet = workbook.add_sheet("benchmark")

worksheet.write(0,0, "batch size")
worksheet.write(0,1, "num heads")
worksheet.write(0,2, "num heads * batch size")
worksheet.write(0,3, "seqlen")
worksheet.write(0,4, "head dim")
worksheet.write(0,5, "run time on A100/ms")
worksheet.write(0,6, "tflops")
worksheet.write(0,7, "util/%")

for i, (batch_size, nheads, seqlen, d) in enumerate(configs):
    index = i + 1
    n = d * nheads  
    flop = (seqlen * seqlen * d * 2 + seqlen * seqlen * d * 2) * batch_size * nheads
    x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

    lengths = torch.randint(seqlen - 20, seqlen, (batch_size, 1), device='cuda')
    attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) < lengths
    attention_mask = torch.zeros(batch_size, seqlen, device='cuda', dtype=dtype)
    attention_mask[~attention_mask_bool] = -10000.0
    attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

    x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask_bool)
    qkv_unpad = rearrange(Wqkv(x_unpad), 'nnz (t h d) -> nnz t h d', t=3,
                        h=nheads).detach().requires_grad_()
    qkv = rearrange(Wqkv(x), 'b s (t h d) -> b s t h d', t=3, h=nheads).detach().requires_grad_()

    fn = lambda qkv_unpad: flash_attn_unpadded_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_seqlen_in_batch, dropout_p, causal=causal
    )

    _, ave_time = benchmark_forward(fn, qkv_unpad, repeats=repeats, desc='FlashAttention')
    ave_time = ave_time.mean * 1000
    tflops = flop / 1e9 / ave_time
    worksheet.write(index, 0, batch_size)
    worksheet.write(index, 1, nheads)
    worksheet.write(index, 2, nheads * batch_size)
    worksheet.write(index, 3, seqlen)
    worksheet.write(index, 4, d)
    worksheet.write(index, 5, ave_time)
    worksheet.write(index, 6, tflops)
    worksheet.write(index, 7, 100 * float(tflops) / DEVICE_TFLOPS)

workbook.save('summary.xls')
# fn = lambda qkv: attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)
# benchmark_all(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')
