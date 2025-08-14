import os
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import config

def generate_story(prompt, save_sample: bool = False, output_dir: str = "artifacts/samples"):
    """
    Generates a story from a given prompt using the fine-tuned model.
    """
    model_dir = config.MODEL_OUTPUT_DIR
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at {model_dir}. Please train the model first.")

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=config.GENERATION_MAX_LENGTH,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
    if save_sample:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sample_file = Path(output_dir) / "sample.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(generated_story)
    return generated_story

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a story from a prompt using the fine-tuned model')
    parser.add_argument('--prompt', type=str, default=config.GENERATION_PROMPT, help='Prompt text')
    parser.add_argument('--save', action='store_true', help='If set, saves a sample to artifacts/samples')
    parser.add_argument('--outdir', type=str, default='artifacts/samples', help='Directory to store generated sample')
    args = parser.parse_args()

    generated_text = generate_story(args.prompt, save_sample=args.save, output_dir=args.outdir)
    print("--- Generated Story ---")
    print(generated_text)