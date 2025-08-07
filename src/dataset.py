import os, sys

sys.path.append(os.getcwd())

import torch
from templates.MSC.peft_cot import peft_cot_template as msc_peft_cot_template
from templates.MSC.peft_label import peft_label_template as msc_peft_label_template
from templates.MASC.peft_cot import peft_cot_template as masc_peft_cot_template
from templates.MASC.peft_label import peft_label_template as masc_peft_label_template
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F


def pad_list_to_max_length(tensor_list, max_len, pad_value):
    # Decoder-only architecture with left-side padding.
    return [F.pad(t, (max_len - t.size(0), 0), value=pad_value) for t in tensor_list]


class FinetuneCoTDataset(Dataset):
    def __init__(self, dataset_key, dataset_type, data, tokenizer, processor):
        self.dataset_key = dataset_key
        self.dataset_type = dataset_type  # train/test
        self.data = data  # 1d list, composed of dicts (e.g. train_json_data)
        self.tokenizer = tokenizer
        self.processor = processor
        self.MAX_LENGTH = 8192  # Set the maximum sequence length.
        self.formatted_texts = self.format_texts()
        self.tokenized_texts = self.tokenize_texts()
        self.raw_answers = self.store_raw_answers()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = {}
        instance["input_ids_c"] = self.tokenized_texts["input_ids_c"][idx]
        instance["attention_mask_c"] = self.tokenized_texts["attention_mask_c"][idx]
        instance["input_ids_l"] = self.tokenized_texts["input_ids_l"][idx]
        instance["attention_mask_l"] = self.tokenized_texts["attention_mask_l"][idx]
        instance["pixel_values"] = self.tokenized_texts["pixel_values"][idx]
        instance["image_grid_thw"] = self.tokenized_texts["image_grid_thw"][idx]

        if self.dataset_type == "train":
            instance["labels_c"] = self.tokenized_texts["labels_c"][idx]
            instance["labels_l"] = self.tokenized_texts["labels_l"][idx]

        instance["id"] = self.data[idx]["id"]
        instance["text"] = self.data[idx]["text"]
        instance["image_path"] = self.data[idx]["image_path"]
        instance["label"] = self.data[idx]["label"]
        instance["cot"] = self.data[idx]["cot"]
        instance["original_index"] = idx
        if self.dataset_key == "T15" or self.dataset_key == "T17":
            instance["aspect"] = self.data[idx]["aspect"]
        return instance

    def format_texts(self):
        # Select a template.
        if self.dataset_key == "SAS" or self.dataset_key == "SAM":
            template_c = msc_peft_cot_template
            template_l = msc_peft_label_template
        elif self.dataset_key == "T15" or self.dataset_key == "T17":
            template_c = masc_peft_cot_template
            template_l = masc_peft_label_template

        formatted_data = dict()
        inputs_c = []
        labels_c = []
        inputs_l = []
        labels_l = []

        for s in self.data:
            current_image_path = os.path.join(
                "Replace here with your path",
                f"data/{self.dataset_key}/Images/{s['image_path']}",
            )
            # CoT
            prompt_c = template_c.format(**s)
            messages_c = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{current_image_path}",
                            "resized_height": 280,
                            "resized_width": 280,
                        },
                        {"type": "text", "text": f"{prompt_c}"},
                    ],
                }
            ]
            inputs_c.append(messages_c)

            if self.dataset_type == "train":
                labels_c.append(f"{s['cot'].strip()}")

            # Label
            prompt_l = template_l.format(**s)
            messages_l = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{current_image_path}",
                            "resized_height": 280,
                            "resized_width": 280,
                        },
                        {"type": "text", "text": f"{prompt_l}"},
                    ],
                }
            ]
            inputs_l.append(messages_l)

            if self.dataset_type == "train":
                labels_l.append(f'"Prediction": {s["label"].strip()}')

        formatted_data["input_c"] = inputs_c
        formatted_data["labels_c"] = labels_c
        formatted_data["input_l"] = inputs_l
        formatted_data["labels_l"] = labels_l

        return formatted_data

    def tokenize_texts(self):
        result = {
            "input_ids_c": [],
            "attention_mask_c": [],
            "input_ids_l": [],
            "attention_mask_l": [],
            "pixel_values": [],
            "image_grid_thw": [],
        }

        if self.dataset_type == "train":
            result["labels_c"] = []
            result["labels_l"] = []

        for idx in range(len(self.data)):
            try:
                # CoT
                single_input_c = [self.formatted_texts["input_c"][idx]]
                text_c = self.processor.apply_chat_template(
                    single_input_c, tokenize=False, add_generation_prompt=True
                )
                image_inputs_c, video_inputs_c = process_vision_info(single_input_c)
                inputs_c = self.processor(
                    text=text_c,
                    images=image_inputs_c,
                    videos=video_inputs_c,
                    padding=True,
                    return_tensors="pt",
                )

                inputs_c = {key: value.tolist() for key, value in inputs_c.items()}
                instruction_c = inputs_c

                if self.dataset_type == "train":
                    response_c = self.tokenizer(
                        f"{self.formatted_texts['labels_c'][idx]}",
                        add_special_tokens=False,
                    )

                    input_ids_c = (
                        instruction_c["input_ids"][0]
                        + response_c["input_ids"]
                        + [self.tokenizer.pad_token_id]
                    )
                    attention_mask_c = (
                        instruction_c["attention_mask"][0]
                        + response_c["attention_mask"]
                        + [1]
                    )

                    label_ids_c = (
                        [-100] * len(instruction_c["input_ids"][0])
                        + response_c["input_ids"]
                        + [self.tokenizer.pad_token_id]
                    )
                else:
                    input_ids_c = instruction_c["input_ids"][0]
                    attention_mask_c = instruction_c["attention_mask"][0]

                # Label
                single_input_l = [self.formatted_texts["input_l"][idx]]

                text_l = self.processor.apply_chat_template(
                    single_input_l, tokenize=False, add_generation_prompt=True
                )

                image_inputs_l, video_inputs_l = process_vision_info(single_input_l)

                inputs_l = self.processor(
                    text=text_l,
                    images=image_inputs_l,
                    videos=video_inputs_l,
                    padding=True,
                    return_tensors="pt",
                )

                inputs_l = {key: value.tolist() for key, value in inputs_l.items()}
                instruction_l = inputs_l

                if self.dataset_type == "train":
                    response_l = self.tokenizer(
                        f"{self.formatted_texts['labels_l'][idx]}",
                        add_special_tokens=False,
                    )

                    input_ids_l = (
                        instruction_l["input_ids"][0]
                        + response_l["input_ids"]
                        + [self.tokenizer.pad_token_id]
                    )
                    attention_mask_l = (
                        instruction_l["attention_mask"][0]
                        + response_l["attention_mask"]
                        + [1]
                    )

                    label_ids_l = (
                        [-100] * len(instruction_l["input_ids"][0])
                        + response_l["input_ids"]
                        + [self.tokenizer.pad_token_id]
                    )
                else:
                    input_ids_l = instruction_l["input_ids"][0]
                    attention_mask_l = instruction_l["attention_mask"][0]

                if len(input_ids_c) > self.MAX_LENGTH:
                    input_ids_c = input_ids_c[: self.MAX_LENGTH]
                    attention_mask_c = attention_mask_c[: self.MAX_LENGTH]
                    if self.dataset_type == "train":
                        label_ids_c = label_ids_c[: self.MAX_LENGTH]

                if len(input_ids_l) > self.MAX_LENGTH:
                    input_ids_l = input_ids_l[: self.MAX_LENGTH]
                    attention_mask_l = attention_mask_l[: self.MAX_LENGTH]
                    if self.dataset_type == "train":
                        label_ids_l = label_ids_l[: self.MAX_LENGTH]

                input_ids_c = torch.tensor(input_ids_c)
                attention_mask_c = torch.tensor(attention_mask_c)
                input_ids_l = torch.tensor(input_ids_l)
                attention_mask_l = torch.tensor(attention_mask_l)

                pixel_values = torch.tensor(inputs_l["pixel_values"])
                image_grid_thw = torch.tensor(inputs_l["image_grid_thw"]).squeeze(0)

                result["input_ids_c"].append(input_ids_c)
                result["attention_mask_c"].append(attention_mask_c)
                result["input_ids_l"].append(input_ids_l)
                result["attention_mask_l"].append(attention_mask_l)
                result["pixel_values"].append(pixel_values)
                result["image_grid_thw"].append(image_grid_thw)

                if self.dataset_type == "train":
                    result["labels_c"].append(torch.tensor(label_ids_c))
                    result["labels_l"].append(torch.tensor(label_ids_l))

            except Exception as e:
                print(f"An error occurred while processing sample {idx}: {str(e)}")
                continue

        # Padding
        # CoT
        max_len_c = max(t.size(0) for t in result["input_ids_c"])
        result["input_ids_c"] = pad_list_to_max_length(
            result["input_ids_c"], max_len_c, pad_value=self.tokenizer.pad_token_id
        )
        result["attention_mask_c"] = pad_list_to_max_length(
            result["attention_mask_c"], max_len_c, pad_value=0
        )

        if self.dataset_type == "train":
            result["labels_c"] = pad_list_to_max_length(
                result["labels_c"], max_len_c, pad_value=-100
            )

        # Label
        max_len_l = max(t.size(0) for t in result["input_ids_l"])
        result["input_ids_l"] = pad_list_to_max_length(
            result["input_ids_l"], max_len_l, pad_value=self.tokenizer.pad_token_id
        )
        result["attention_mask_l"] = pad_list_to_max_length(
            result["attention_mask_l"], max_len_l, pad_value=0
        )

        if self.dataset_type == "train":
            result["labels_l"] = pad_list_to_max_length(
                result["labels_l"], max_len_l, pad_value=-100
            )

        return result

    def store_raw_answers(self):
        # Store raw answers for evaluation or other purposes.
        raw_answers = [s["label"] for s in self.data]
        return raw_answers
