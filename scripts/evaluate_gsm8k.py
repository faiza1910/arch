import argparse
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_final_number(text):
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else "NaN"

def generate_completions(model, tokenizer, prompt, k=7, max_new_tokens=128, temperature=0.0, device='cuda'):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    completions = []
    for i in range(outputs.shape[0]):
        text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        generated_text = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        completions.append(extract_final_number(generated_text))
    return completions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained model & tokenizer")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to GSM8K test CSV")
    parser.add_argument("--output_csv", type=str, default="model_outputs.csv", help="Path to save completions")
    parser.add_argument("--k", type=int, default=5, help="How many completions per question (pass@K)")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"Loading test set from {args.test_csv}")
    df = pd.read_csv(args.test_csv)

    rows = []
    print(f"Generating {args.k} completions per question...")
    for idx, row in df.iterrows():
        question = row["question"].strip()
        gold_full = row["answer"].strip()
        gold = extract_final_number(gold_full)
        prompt = question + " Answer:"
        completions = generate_completions(
            model, tokenizer, prompt, k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )
        row_dict = {
            "question": question,
            "gold": gold,
        }
        for i, comp in enumerate(completions):
            row_dict[f"pred_{i+1}"] = comp
        rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved completions to {args.output_csv}")

if __name__ == "__main__":
    main()
