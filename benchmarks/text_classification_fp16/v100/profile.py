from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from papaya import PapayaProfiler, local_sharing

class BertOrgProfiler(PapayaProfiler):
    @local_sharing
    def init(self):
        model_name = "bert-large-cased"
        padding = "max_length"
        max_length = 128
        dataset = "glue"
        learning_rate = 5e-5
        weight_decay = 0.01
        metric = load_metric("accuracy")
        lr_scheduler_type = "linear"

        set_seed(0)
        accelerator = Accelerator(fp16=True, device_placement=False)
        accelerator.wait_for_everyone()

        raw_datasets = load_dataset(dataset)
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
        )

        # Preprocessing the datasets
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
        label_to_id = {v: i for i, v in enumerate(label_list)}
        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        train_max_length = 0
        dev_max_length = 0
        for item in train_dataset:
            if len(item['input_ids']) > train_max_length:
                train_max_length = len(item['input_ids'])
        for item in eval_dataset:
            if len(item['input_ids']) > dev_max_length:
                dev_max_length = len(item['input_ids'])
        
         # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
        )


    def create_dataloader(self, batch_size):
        data_collator = default_data_collator
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    def run_iter(self, batch):
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        lr_scheduler.step()



if __name__ == "__main__":

    p = BertOrgProfiler()
    p.profile()
    p.predict_max_tpt()



