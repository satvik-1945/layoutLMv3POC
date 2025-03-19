import numpy as np
import os
from datasets import load_dataset, load_metric, Features, Array3D, Sequence, Array2D, Value
from transformers import LayoutLMv3ForTokenClassification, TrainingArguments, Trainer, LayoutLMv3Processor
from transformers.data.data_collator import default_data_collator

base_model = 'microsoft/layoutlmv3-large'
label_list = ['B-ANSWER_RADIO', 'I-ANSWER_RADIO', 'E-ANSWER_RADIO', 'B-ANSWER_TEXT', 'I-ANSWER_TEXT', 'E-ANSWER_TEXT',
              'B-QUESTION', 'I-QUESTION', 'E-QUESTION', 'B-TABLE', 'I-TABLE', 'E-TABLE',
              'B-OTHERS', 'I-OTHERS', 'E-OTHERS']
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}
model = LayoutLMv3ForTokenClassification.from_pretrained(base_model, num_labels=len(label_list),
                                                         id2label=id2label, label2id=label2id)
processor = LayoutLMv3Processor.from_pretrained(base_model, apply_ocr=False)


def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def prepare_examples(examples):
    images = examples["image"]
    words = examples["tokens"]
    boxes = examples["bboxes"]
    word_labels = examples["ner_tags"]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        stride=128,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        return_offsets_mapping=True)

    encoding.pop('offset_mapping')
    encoding.pop('overflow_to_sample_mapping')
    return encoding


def train():
    dataset = load_dataset("DatasetBuilder.py")
    print(dataset)
    column_names = dataset["train"].column_names
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })
    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
        batch_size=10,
        num_proc=os.cpu_count()
    )
    eval_dataset = dataset["test"].map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
        batch_size=10,
        num_proc=os.cpu_count()
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir='../../../models',
        max_steps=10,
        eval_steps=2,
        save_steps=4,
        learning_rate=1e-1,
        evaluation_strategy='steps',
        metric_for_best_model='accuracy',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(os.environ['SM_MODEL_DIR']) #save to sagemaker model dir



train()