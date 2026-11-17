#!/usr/bin/env python3
# kvattn_gen.py <repo> <rev|WORK> <out.inc>: the decode attention kernels of one engine source, cut verbatim out of qwen38_mk.cu
# for bench/kvattn_bench.cu (which includes two of them in separate namespaces: base and working tree, compared bit for bit).
import re
import subprocess
import sys

repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
if rev == 'WORK':
    src = open(repo + '/qwen38_mk.cu', encoding='utf-8').read()
else:
    src = subprocess.run(['git', '-C', repo, 'show', rev + ':qwen38_mk.cu'], capture_output=True, text=True, check=True).stdout
lines = src.split('\n')

# every function of the attention path, picked by name at column 0 (with its template line); new helpers: prefix aqx_ / kvx_
NAMES = '''wsum tid_x cp_async16 load8 load8_ro rope8 rope_row mma16816 aq_i8_q aq_i8_q0 aq_i8_acc mma_s8z aq_i8_s
kv_store_q8 kv_store_q4 fwht256 kv_store_qn kv_store kv_ws kv_wsm kv_fmt kv_ring_tile kv_tile_src
kv_spread4 kv_rows_expand kv_tile_expand attn_dims mtp_ws attn_dims_w attn_chunk att_rank
aq_q8x2 aq_q8x2p aq_sub aq_issue attn_core_q8_t attn_core_q8 attn_core_q8m kv_slice_expand attn_core_w48
attn_core_w48q attn_core_w48m'''.split()
PAT = re.compile(r'^__device__ .*?\b(' + '|'.join(NAMES) + r'|aqx_\w+|kvx_\w+)\s*\(')
ONE = re.compile(r'^(constexpr (int|float) (SM_ATTN_W48|AQ_I8_QS|AQ_I8_LDQ|AQX_\w+)\b|static_assert\(SM_ATTN_W48|struct KvFmt\b)')
res = []
i = 0
seen = set()
while i < len(lines):
    ln = lines[i]
    if ONE.match(ln):
        res.append(ln)
        i += 1
        continue
    m = PAT.match(ln)
    if not m:
        i += 1
        continue
    seen.add(m.group(1))
    s = i - 1 if i > 0 and lines[i - 1].startswith('template') else i
    depth, j, opened = 0, i, False
    pp = [] # #else branches do not count: both branches of an #if may open the same block
    while j < len(lines):
        d = lines[j].lstrip()
        if d.startswith('#if'):
            pp.append(False)
        elif d.startswith('#else'):
            pp[-1] = True
        elif d.startswith('#endif'):
            pp.pop()
        elif not any(pp):
            depth += lines[j].count('{') - lines[j].count('}')
            opened = opened or '{' in lines[j]
        if opened and depth == 0:
            break
        if not opened and lines[j].rstrip().endswith(';'):
            break
        j += 1
    res.extend(lines[s:j + 1])
    res.append('')
    i = j + 1
missing = [n for n in NAMES if n not in seen]
if missing:
    sys.exit('kvattn_gen: not found: ' + ' '.join(missing))
open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(res) + '\n')
print('%s: %d lines from %s' % (out, len(res), rev))
