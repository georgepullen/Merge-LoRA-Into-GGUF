import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM
import convert

def merge_and_save(model_name, lora, base, format):
    base_model = AutoModelForCausalLM.from_pretrained(base)
    peft_model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=lora)
    merged_weights = peft_model.merge_and_unload()
    merged_weights.save_pretrained(f"{model_name}-hf")

def convert_to_ggml(model_name, base, format):
    convert.main([
        f"{model_name}-hf",
        "--vocab-dir", {base},
        "--outfile", f"{model_name}.{format}.bin",
        "--outtype", format
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge, save, and convert PEFT model weights")
    parser.add_argument("model_name", help="Name of your model")
    parser.add_argument("-l", "--lora", required=True, help="Path to the low rank adapter")
    parser.add_argument("-b", "--base", required=True, help="Path to the base model checkpoint")
    parser.add_argument("-f", "--format", default="f16", help="F32, F16, q4_0, etc... (default: f16)")
    args = parser.parse_args()

    model_dir = merge_and_save(args.model_name, args.lora, args.base, args.format)
    convert_to_ggml(args.model_name, args.base, args.format)
    print(f"{args.model_name} successfully merged, saved, and converted to {args.format} ggml format.")
