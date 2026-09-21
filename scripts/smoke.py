import torch, json, os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
MP = "/user/yac/LinearSwap/models/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MP)
cfg = AutoConfig.from_pretrained(MP)
print("layer_types:", cfg.text_config.layer_types)
m = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map=None)
print(type(m).__name__)
m = m.to("cuda:0").eval()
p = "Dictionary: apple=3, banana=7, cherry=1, date=9.\nQuestion: What is the value of cherry?\nAnswer:"
ids = tok(p, return_tensors="pt").input_ids.cuda()
print("ntok", ids.shape)
with torch.no_grad():
    out = m(ids)
lg = out.logits[0,-1]
top = lg.topk(5)
print([(tok.decode([i]), round(v.item(),2)) for i,v in zip(top.indices, top.values)])
print("tokens:", [tok.decode([t]) for t in ids[0].tolist()])
