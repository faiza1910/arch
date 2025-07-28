import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the Huggingface model to use")
parser.add_argument("--train_csv", type=str, default="data/gsm8k_train.csv", help="Path to train csv")
parser.add_argument("--test_csv", type=str, default="data/gsm8k_test.csv", help="Path to test csv")
parser.add_argument("--output_dir", type=str, default="./results", help="Where to save model/results")
parser.add_argument("--logs_dir", type=str, default="./logs", help="Where to save logs")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
parser.add_argument("--gpus", type=str, default="all", help="Which GPUs to use. Set to 'all' for multi-GPU, or a comma-separated list")
args = parser.parse_args()

if args.gpus != "all":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader


if __name__ == "__main__": 
    dataset = load_dataset(
        'csv',
        data_files={
            'train': args.train_csv,
            'test': args.test_csv
        }
    )

    #print("Loaded splits:", dataset)
    #print("Sample train example:", dataset['train'][0])
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    IGNORE_IDX = -100
    def tokenize(batch):
        prompt = batch["question"].strip() + " Answer: "
        answer = batch["answer"].strip()
        text = prompt + " " + answer 
        tokens = tokenizer(
            text,
            padding = "max_length",
            truncation = True,
            max_length = 128,
        )
        prompt_ids = tokenizer(prompt, max_length=128, truncation=True)["input_ids"]
        num_prompt = len(prompt_ids)

        labels = [IGNORE_IDX]*num_prompt + tokens["input_ids"][num_prompt:]
        labels = labels[:128]
        tokens["labels"] = labels
        return tokens
    tokenized_dataset = dataset.map(tokenize, batched=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        learning_rate = args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_dir = args.logs_dir,
        #evaluation_strategy = "epoch",
        
    )

    training_args = training_args.set_logging(
        strategy = "steps",
        steps = 100,
        level="info",
        report_to="wandb"
    )


    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset= tokenized_dataset["train"],
        eval_dataset= tokenized_dataset["test"]
    )

    trainer.train()

    #metrics = trainer.evaluate(tokenized_dataset["test"])
    #print("Test set metrics:", metrics)
    #print("Test set loss:", metrics.get("eval_loss"))
    model.save_pretrained(f"{args.output_dir}/model")
    tokenizer.save_pretrained(f"{args.output_dir}/model")