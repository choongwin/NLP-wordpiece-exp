from tokenizers import Tokenizer, models, trainers,pre_tokenizers,decoders
import json
import torch
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BertTokenizerFast,
    BertModel,
)
import evaluate
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = evaluate.load("f1").compute(predictions=predictions, references=labels, average="macro")["f1"]
    f1_micro = evaluate.load("f1").compute(predictions=predictions, references=labels, average="micro")["f1"]

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bio_tokenizer = Tokenizer.from_file("biometical_tokenizer_50000.json")

# 筛选
bio_vocab = bio_tokenizer.get_vocab()
bert_vocab = bert_tokenizer.get_vocab()

numbers = {'0','1','2','3','4','5','6','7','8','9'}
unique_bio_tokens = [
    token for token in bio_vocab
    if (token not in bert_vocab) and
       (not token.startswith('#')) and
       (not token[-1] in numbers) and
       (len(token) >= 6)
]

selected_tokens = sorted(unique_bio_tokens, key=lambda x: bio_vocab[x], reverse=True)[:5000]

# 加入
bert_tokenizer.add_tokens(selected_tokens)
bert_tokenizer.save_pretrained("expanded_bert_tokenizer")

# resize
model = BertModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(bert_tokenizer))
with torch.no_grad():
    mean_embedding = model.embeddings.word_embeddings.weight[:len(bert_vocab)].mean(dim=0)
    model.embeddings.word_embeddings.weight[len(bert_vocab):] = mean_embedding

model.save_pretrained("expanded_bert_model")

# 训练
# 读入
test_df = pq.read_table("Hoc/test.parquet").to_pandas()
train_df = pq.read_table("Hoc/train.parquet").to_pandas()

hoc_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

label_num = 11
tokenizer = BertTokenizerFast.from_pretrained("expanded_bert_tokenizer")
model = BertForSequenceClassification.from_pretrained(
    "expanded_bert_model",
    num_labels=label_num
)

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

for param in model.base_model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if any(keyword in name for keyword in [
        "LayerNorm.weight", "LayerNorm.bias",
        "encoder.layer.11.",
        "pooler.dense.weight", "pooler.dense.bias"
    ]):
        param.requires_grad = True

tokenized_data = hoc_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("suceed")

training_args = TrainingArguments(
    output_dir="./result",
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()