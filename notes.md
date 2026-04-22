

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct

uv run python web_demo_mm.py --backend vllm --quantization fp8 -c ./Qwen3-VL-8B-Instruct
```