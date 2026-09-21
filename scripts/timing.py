import sys, time, torch; sys.path.insert(0,'/user/yac/LinearAblation')
from src.task import make_examples
from src.runner import Runner
for dt in (torch.bfloat16, torch.float32):
    R=Runner(dtype=dt)
    exs=make_examples(25,8,seed=3,target_pos=2,distract_pos=5)
    ids=R.tok([e.clean_prompt() for e in exs],return_tensors="pt").input_ids.cuda()
    print(dt, ids.shape)
    for _ in range(2): R.forward(ids)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(10): R.forward(ids)
    torch.cuda.synchronize(); print("  per batched fwd: %.3fs"%((time.time()-t)/10))
    del R; torch.cuda.empty_cache()
