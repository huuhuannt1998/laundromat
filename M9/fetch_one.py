import os, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download, HfApi
r = sys.argv[1]
files = [f.rfilename for f in HfApi().model_info(r).siblings]
fmt = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
print(f"--- {r} [{fmt}]", flush=True)
snapshot_download(r, allow_patterns=["*.json","*.txt","*.model",fmt], max_workers=4)
print("DONE-FETCH")
