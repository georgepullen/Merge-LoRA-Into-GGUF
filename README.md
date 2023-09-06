# Merge LoRA, convert to GGUF, and quantize.
A time-saving script for any Llama CPP/LoRA workflow: it handles merging the LoRA into the base model, converting it to GGUF format, and applying post-training quantization.

## Setup
* Clone this repository: ```git clone https://github.com/georgepullen/Merge-LoRA-Into-GGUF.git .```
* Clone the llama.cpp repository: ```git clone https://github.com/ggerganov/llama.cpp.git .```
* Install requirements for both repositories: ```pip install -r requirements.txt```, ```pip install -r merge_lora_requirements.txt```

## Arguments
* ```-l``` Path to LoRA (HF Repo ID or Local Path)
* ```-b``` Path to base model (HF Repo ID or Local Path)
* ```-f``` Format (F32, F16, q6_k, q4_0, etc.)
