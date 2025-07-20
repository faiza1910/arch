import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the Huggingface model to use")
parser.add_argument("--train_csv", type=str, default="../data/gsm8k_train.csv", help="Path to train csv")
parser.add_argument("--test_csv", type=str, default="../data/gsm8k_test.csv", help="Path to test csv")
parser.add_argument("--output_dir", type=str, default="./results", help="Where to save model/results")
parser.add_argument("--logs_dir", type=str, default="./logs", help="Where to save logs")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
args = parser.parse_args()


dataset = load_dataset(
    'csv',
    data_files={
        'train': args.train_csv,
        'test': args.test_csv
    }
)

print("Loaded splits:", dataset)
print("Sample train example:", dataset['train'][0])
 
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    text = batch["question"] + " Answer: " + batch["answer"]
    tokens = tokenizer(
        text,
        padding = "max_length",
        truncation = True,
        max_length = 128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
tokenized_dataset = dataset.map(tokenize, batched=False)

#tokenized_dataset.set_format(type='torch', columns=['input_ids','attention_mask'])

#train_loader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True)
#test_loader = DataLoader(tokenized_dataset['test'], batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =  AutoModelForCausalLM.from_pretrained(args.model_name)
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    logging_dir = args.logs_dir,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= tokenized_dataset["train"],
    eval_dataset= tokenized_dataset["test"]
)

trainer.train()

model.save_pretrained(f"{args.output_dir}/model")
tokenizer.save_pretrained(f"{args.output_dir}/model")