import numpy as np, json, sys
# usage: python make_palette.py tid2eid_0731.npz [n=64] [cap=3] [max_id=128000]
z = np.load(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 3          # max times any expert may be hit per layer
max_id = int(sys.argv[4]) if len(sys.argv) > 4 else 128000   # exclude special/added IDs
tabs = [z[k] for k in sorted(z.files)]; V = tabs[0].shape[0]
M = []
for t in tabs:
    m = np.zeros((V, 256), dtype=np.int16); m[np.arange(V)[:, None], t] = 1; M.append(m[:max_id])
cnt = [np.zeros(256, np.int16) for _ in tabs]; picks = []
for _ in range(n):
    g = np.zeros(max_id, np.int32); ok = np.ones(max_id, bool)
    for m, c in zip(M, cnt):
        g += (m * (c == 0)).sum(1)                    # new experts covered
        ok &= ((m * (c >= cap)).sum(1) == 0)          # would exceed the cap
    g[~ok] = -1
    for p in picks: g[p] = -1
    j = int(np.argmax(g)); picks.append(j)            # tie-break: lowest ID
    for m, c in zip(M, cnt): c += m[j]
print(json.dumps(picks))
print('coverage:', [int((c > 0).sum()) for c in cnt], 'max multiplicity:', [int(c.max()) for c in cnt], file=sys.stderr)
