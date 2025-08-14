import argparse
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer
import config

def load_and_prepare_data(data_file: str | None = None):
    """
    Loads data from a CSV file, renames columns, and prepares it for tokenization.
    """
    source_file = data_file or config.DATA_FILE
    dataset = load_dataset('csv', data_files=source_file)
    
    if 'text' in dataset['train'].column_names:
        dataset = dataset['train'].remove_columns(['text'])
    else:
        dataset = dataset['train']
    
    dataset = dataset.rename_column(config.TEXT_COLUMN, config.TARGET_COLUMN)
    return dataset

def tokenize_data(dataset):
    """
    Tokenizes the dataset using GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        input_ids = tokenizer(
            examples[config.TARGET_COLUMN],
            padding='max_length',
            truncation=True,
            max_length=config.MAX_LENGTH,
        )
        input_ids['labels'] = input_ids['input_ids'].copy()
        return input_ids

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

def split_data(tokenized_datasets):
    """
    Splits the tokenized dataset into training and validation sets.
    """
    train_test_split = tokenized_datasets.train_test_split(test_size=config.TEST_SIZE)
    final_datasets = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })
    return final_datasets

def get_processed_data(data_file: str | None = None):
    """
    Main function to load, process, and split the data.
    """
    dataset = load_and_prepare_data(data_file=data_file)
    tokenized_datasets = tokenize_data(dataset)
    final_datasets = split_data(tokenized_datasets)
    return final_datasets

def save_processed_to_disk(output_dir: str, data_file: str | None = None) -> None:
    datasets = get_processed_data(data_file=data_file)
    datasets.save_to_disk(output_dir)

def preprocess_csv(input_csv: str, output_csv: str) -> None:
    """
    Minimal preprocessing to ensure a column named config.TEXT_COLUMN exists in the output CSV.
    If the input already contains that column it will be preserved; otherwise we copy from
    a likely raw column named 'text'.
    """
    df = pd.read_csv(input_csv)
    if config.TEXT_COLUMN in df.columns:
        cleaned = df[[config.TEXT_COLUMN]].copy()
    elif 'text' in df.columns:
        cleaned = df[['text']].copy()
        cleaned.rename(columns={'text': config.TEXT_COLUMN}, inplace=True)
    else:
        # fallback: take the first text-like column
        text_like_cols = [c for c in df.columns if 'text' in c.lower()]
        if not text_like_cols:
            raise ValueError(f"No text column found in {input_csv}. Columns: {df.columns.tolist()}")
        cleaned = df[[text_like_cols[0]]].copy()
        cleaned.rename(columns={text_like_cols[0]: config.TEXT_COLUMN}, inplace=True)
    cleaned.to_csv(output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data processing utilities")
    parser.add_argument('--input', type=str, default=None, help='Optional path to input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Optional output directory to save tokenized DatasetDict')
    parser.add_argument('--csv_out', type=str, default=None, help='If provided, write a cleaned CSV to this path')
    args = parser.parse_args()

    if args.csv_out:
        if not args.input:
            raise SystemExit("--csv_out requires --input to be set")
        preprocess_csv(args.input, args.csv_out)
        print(f"Cleaned CSV written to {args.csv_out}")
    elif args.output:
        save_processed_to_disk(output_dir=args.output, data_file=args.input)
        print(f"Processed datasets saved to {args.output}")
    else:
        processed_data = get_processed_data(data_file=args.input)
        print(processed_data)
        print("Train dataset size:", len(processed_data['train']))
        print("Validation dataset size:", len(processed_data['validation']))