import argparse
from model import train_model
from inference import generate_story
import config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for story generation.")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'generate'], 
        required=True,
        help="Choose 'train' to fine-tune the model or 'generate' to create a story."
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=config.GENERATION_PROMPT,
        help="The prompt to use for story generation (only in 'generate' mode)."
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("--- Starting Model Training ---")
        train_model()
        print("--- Model Training Finished ---")
    
    elif args.mode == 'generate':
        print("--- Starting Story Generation ---")
        generated_text = generate_story(args.prompt, save_sample=True)
        print("\n--- Generated Story ---")
        print(generated_text)
        print("\n--- Story Generation Finished ---")

if __name__ == '__main__':
    main() 