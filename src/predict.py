import argparse
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)
import wandb
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate

# Log in to Weights & Biases using your API key

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="../train_kan.pickle")  # Path to your pickle file
parser.add_argument("--eval_data", type=str, default="../test_kan.pickle")  # Path to your pickle file
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--label_all_tokens", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--add_lang_tag", type=str, default=None)
parser.add_argument("--og_lang", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

def preprocess_function(examples):
    return tokenizer(examples["text"],
                     truncation=True,
                     padding="max_length",
                     max_length=args.max_seq_length)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Load training data from pickle file
train_df = pd.read_pickle(args.train_data)
eval_df = pd.read_pickle(args.eval_data)
len(eval_df)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

if args.add_lang_tag:
    train_dataset = train_dataset.map(lambda x: {"text": f"<{args.add_lang_tag}> {x['text']}"})
    eval_dataset = eval_dataset.map(lambda x: {"text": f"<{args.add_lang_tag}> {x['text']}"})

label_list = train_dataset.unique('label')

metric = load_metric('glue', 'sst2', trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_list))

if args.do_train:
    # Training logic as before...
    pass

if args.do_predict:
    # Preprocess the evaluation dataset
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    print(f"Length of Validation Dataset: {len(eval_dataset)}")

    # Make predictions
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.eval_batch_size,
            fp16=args.fp16,
            do_train=False,
            do_eval=True,
        ),
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(eval_dataset)
    print(len(predictions))
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Write predictions to a text file
    with open("predictions.txt", 'w') as f:
        for text, label in zip(eval_df["text"][:len(predicted_labels)], predicted_labels):
            if "\n" in text:
                text = text.replace("\n"," ")
            f.write(f"{text}\t{label}\n")  # Write each text and its predicted label

    print("Predictions saved to predictions.txt")


# Optional: Your wandb sweep configuration here...
