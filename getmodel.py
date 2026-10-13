import numpy as np, torch, os, struct
os.makedirs("golden", exist_ok=True)
model.eval()

def save(name, t):
    t.detach().cpu().numpy().astype(np.float32).tofile(f"golden/{name}.bin")

def make_hook(name):
    def hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out   # MHA trả tuple
        save(name + ".out", o)
    return hook

for n, m in model.named_modules():
    if n: m.register_forward_hook(make_hook(n))

x = preprocess_giong_luc_train("0010.png")      # [1,3,32,32]
save("input", x)
with torch.no_grad():
    logits = model(x)
save("logits", logits)

# weights theo đúng thứ tự state_dict
with open("weights.bin", "wb") as f:
    for k, v in model.state_dict().items():
        f.write(v.detach().cpu().numpy().astype(np.float32).tobytes())