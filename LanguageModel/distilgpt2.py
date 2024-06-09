from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

giga = load_dataset('text', data_files={'train':'data/gigaspeech/train.txt',
                                        'validation': 'data/gigaspeech/valid.txt',
                                        'test': 'data/gigaspeech/test.txt'})


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    return tokenizer([s + ' <eos>' for s in examples["text"]], truncation=True)


# print(giga["train"].column_names)
tokenized_giga = giga.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=giga["train"].column_names,
)

block_size = 128

# print(tokenized_giga)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # for k in concatenated_examples.keys():
    #     concatenated_examples[k] += [tokenizer.pad_token] * int(len(concatenated_examples[k]) % block_size)
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i * block_size : (i+1) * block_size] for i in range(0, total_length // block_size - 1)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_giga.map(group_texts, batched=True, num_proc=4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="exp/distilgpt2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=5.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
