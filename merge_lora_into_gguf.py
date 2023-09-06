from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM
import argparse
import convert
import os

def merge_and_save(model_name, lora, base):
    base_model = AutoModelForCausalLM.from_pretrained(base)
    peft_model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=lora)
    merged_weights = peft_model.merge_and_unload()
    merged_weights.save_pretrained(f"{model_name}-hf")

def convert_to_gguf(model_name, base):
    convert.main([
        f"{model_name}-hf",
        "--vocab-dir", base,
        "--outfile", f"{model_name}.gguf",
    ])

def apply_quantization(model_name, quant_format):
    os.system(f"./quantize {model_name}.gguf {model_name}.{quant_format}.gguf {quant_format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge, save, and convert PEFT model weights")
    parser.add_argument("model_name", help="Name of your model")
    parser.add_argument("-l", "--lora", required=True, help="Path to the low rank adapter")
    parser.add_argument("-b", "--base", required=True, help="Path to the base model checkpoint")
    parser.add_argument("-q", "--quantize", default="none", help="q8_0, q6_K, q4_K_M etc... (default: none)")
    args = parser.parse_args()

    model_dir = merge_and_save(args.model_name, args.lora, args.base)
    convert_to_gguf(args.model_name, args.base)
    if args.quantize != "none":
        apply_quantization(args.model_name, args.quantize)

    print(f"{args.model_name} successfully merged, saved, and converted to {args.format} GGUF format.")
