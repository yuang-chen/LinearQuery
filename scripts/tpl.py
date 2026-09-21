import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MP="/user/yac/LinearSwap/models/Qwen3.5-0.8B"
tok=AutoTokenizer.from_pretrained(MP)
m=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16).cuda().eval()
tpls=[
 "Dictionary: apple=3, banana=7, cherry=1, date=9.\nQuestion: What is the value of cherry?\nAnswer: ",
 "Dictionary: apple=3, banana=7, cherry=1, date=9.\nQuestion: What is the value of cherry?\nAnswer: cherry=",
 "apple=3\nbanana=7\ncherry=1\ndate=9\n\nQ: cherry\nA:",
 "apple: 3\nbanana: 7\ncherry: 1\ndate: 9\n\nWhat is cherry? It is",
]
for p in tpls:
    ids=tok(p,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): lg=m(ids).logits[0,-1]
    t=lg.topk(5)
    print(repr(p[-30:]), "->", [(tok.decode([i]),round(v.item(),1)) for i,v in zip(t.indices,t.values)])
    print("  last toks:", [tok.decode([x]) for x in ids[0,-6:].tolist()])
