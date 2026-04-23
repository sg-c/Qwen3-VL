
```bash
pip install uv

git clone https://github.com/sg-c/Qwen3-VL.git

cd Qwen3-VL
uv venv
source .venv/bin/activate
uv pip install -r requirements_web_demo.txt
uv pip install qwen-vl-utils vllm

hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct

# uv run python web_demo_mm.py \
# --backend vllm \
# --quantization fp8 \
# -c ./Qwen3-VL-8B-Instruct \
# --max-model-len 8192 \
# --gpu-memory-utilization 0.9 \
# --server-name 0.0.0.0 \
# --server-port 7860


vllm serve ./Qwen3-VL-8B-Instruct \
  --trust-remote-code \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --host 0.0.0.0 \
  --port 7860
```