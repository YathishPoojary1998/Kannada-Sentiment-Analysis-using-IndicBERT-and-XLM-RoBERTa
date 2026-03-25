import argparse
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)
import wandb
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate


# Log in to Weights & Biases using your API key
wandb.login(key="6b8b21bf5e13170c36c352edd54624a7b5a8c10f")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="../train_kan.pickle")  # Path to your pickle file
parser.add_argument("--eval_data", type=str, default="../valid_kan.pickle")  # Path to your pickle file
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

def train(config=None):

    wandb.init(config=config)
    config = wandb.config
    wandb.run.name = f"{args.model_name}-{args.train_data}-lr-{config.lr}-wd-{config.weight_decay}"
    wandb.run.save()

    def preprocess_function(examples):
        return tokenizer(examples["text"],
                         truncation=True,
                         padding="max_length",
                         max_length=args.max_seq_length
                         )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Load training data from pickle file
    train_df = pd.read_pickle(args.train_data)
    eval_df = pd.read_pickle(args.eval_data)

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
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        eval_dataset = eval_dataset.map(preprocess_function, batched=True)

        print(f"Length of Training Dataset: {len(train_dataset)}")
        print(f"Length of Validation Dataset: {len(eval_dataset)}")

        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}-{str(config.lr)}-{str(config.weight_decay)}",
            save_total_limit=5,
            save_strategy="steps",
            learning_rate=config.lr,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            do_eval=True,
            evaluation_strategy="steps",
            weight_decay=config.weight_decay,
            fp16=args.fp16,
            warmup_ratio=args.warmup_ratio,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        trainer.evaluate()

if args.do_train:
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'lr': {'values': [3e-5]},
            'weight_decay': {'values': [0.01]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="huggingface", entity="indic-indic")
    wandb.agent(sweep_id, train)
