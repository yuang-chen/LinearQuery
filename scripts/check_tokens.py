import sys; sys.path.insert(0,'/user/yac/LinearAblation')
from transformers import AutoTokenizer
from src.task import *
tok=AutoTokenizer.from_pretrained(MODEL_PATH)
bad_k=[k for k in KEY_WORDS if len(tok(" "+k).input_ids)!=1]
bad_v=[v for v in VALUE_WORDS if len(tok(v).input_ids)!=1]
print("multi-token keys:",bad_k); print("multi-token values:",bad_v)
print("overlap:", set(KEY_WORDS)&set(VALUE_WORDS))
for npair in (4,8,16):
    exs=make_examples(100,npair,seed=1)
    res=[verify_single_token(tok,e) for e in exs]
    ok=sum(r[0] for r in res)
    print(f"n_pairs={npair}: {ok}/100 verified; ntok={len(tok(exs[0].clean_prompt()).input_ids)}")
    if ok<100: print("  fail:",[r[1] for r in res if not r[0]][:3])
print(repr(exs[0].clean_prompt())); print(repr(exs[0].corrupt_prompt()))
exs=make_examples(2,4,seed=1,template='alt'); print(repr(exs[0].clean_prompt()))
print(verify_single_token(tok,exs[0]))
