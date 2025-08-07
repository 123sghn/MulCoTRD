import os, sys

sys.path.append(os.getcwd())
import random
import torch
import numpy as np
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model, TaskType

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from sklearn.metrics import f1_score, confusion_matrix

from bleu.bleu import Bleu
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from rouge import Rouge
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.translate.meteor_score import meteor_score

tree_tokenizer = TreebankWordTokenizer()


def get_model_and_tokenizer(model_size, device):
    print("Start loading the model...")
    model_name = f"Replace here with your path/Qwen2.5-VL-{model_size}-Instruct/"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True, padding_side="left"
    )
    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

    # model.enable_input_require_grads()  # This method should be executed when gradient checkpointing is enabled.

    # LoRA Config
    if model_size == "7B":
        lora_dropout = 0.2
    else:
        lora_dropout = 0.05
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # Training mode.
        r=16,
        lora_alpha=32,  # Lora alpha
        lora_dropout=lora_dropout,  # Dropout rate for LoRA layers.
        bias="none",
    )

    # Get LoRA model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    print("Model loading complete!")

    return model, tokenizer, processor


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def return_evaluations_in_boolean(
    evaluator, raw_pred, raw_ans, return_cleansed_predictions=False
):
    c_pred = [evaluator.cleanse_prediction(pred) for pred in raw_pred]  # list[str]
    c_answ = [evaluator.cleanse_answer(answer) for answer in raw_ans]  # list[str]
    assert len(c_answ) == len(
        c_pred
    ), f"Prediction: {len(c_pred)}, Answer: {len(c_answ)} does not match!"

    evaluations = [
        evaluator._compare_prediction_and_answer(pred, ans)
        for pred, ans in zip(c_pred, c_answ)
    ]

    weighted_f1 = f1_score(
        c_answ, c_pred, average="weighted", labels=["positive", "neutral", "negative"]
    )
    macro_f1 = f1_score(
        c_answ, c_pred, average="macro", labels=["positive", "neutral", "negative"]
    )
    conf_matrix = confusion_matrix(
        c_answ, c_pred, labels=["positive", "neutral", "negative"]
    )

    if return_cleansed_predictions:
        return evaluations, c_pred, weighted_f1, macro_f1, conf_matrix
    else:
        return evaluations, weighted_f1, macro_f1, conf_matrix


# Compute the similarity score between `cot_gt` and `cot_pre`.
def calculate_pairwise_bleu_avg(predictions, references):

    assert len(predictions) == len(
        references
    ), "The number of predictions does not match the number of references"

    bleu = Bleu()
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        result = bleu.compute(predictions=[pred], references=[[ref]])
        bleu_scores.append(result["bleu"])

    avg_bleu = np.mean(bleu_scores)
    return avg_bleu


# Compute the sentence-level one-to-one similarity between two lists of strings using average cosine similarity
def sentence_transformers_similarity_batch(predictions, references):
    assert len(predictions) == len(
        references
    ), "The lengths of predictions and references must be equal (one-to-one correspondence required)"

    model = SentenceTransformer("Replace here with your path")

    embeddings_A = model.encode(
        predictions, convert_to_numpy=True, show_progress_bar=False
    )
    embeddings_B = model.encode(
        references, convert_to_numpy=True, show_progress_bar=False
    )

    cosine_similarities = [
        1 - cosine(vec1, vec2) for vec1, vec2 in zip(embeddings_A, embeddings_B)
    ]

    average_score = np.mean(cosine_similarities)
    return average_score


# Ensure necessary NLTK data is downloaded (required on first run).
# nltk.download('punkt')
# nltk.download('wordnet')


def calculate_pairwise_meteor_avg(predictions, references):
    assert len(predictions) == len(
        references
    ), "The number of predictions does not match the number of references"

    meteor_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = tree_tokenizer.tokenize(pred.lower())
        ref_tokens = tree_tokenizer.tokenize(ref.lower())

        score = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(score)

    avg_meteor = np.mean(meteor_scores)
    return avg_meteor


def calculate_pairwise_rouge_l_avg(predictions, references):
    assert len(predictions) == len(
        references
    ), "The number of predictions does not match the number of references"

    rouge = Rouge()
    rouge_l_scores = []

    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            rouge_l_scores.append(0.0)
            continue

        try:
            scores = rouge.get_scores(pred, ref)
            rouge_l_f1 = scores[0]["rouge-l"]["f"]
            rouge_l_scores.append(rouge_l_f1)
        except:
            rouge_l_scores.append(0.0)

    avg_rouge_l = np.mean(rouge_l_scores)
    return avg_rouge_l


def calculate_distinct_n(texts, n=2):
    if not texts:
        return 0.0

    distinct_scores = []

    for text in texts:
        tokens = tree_tokenizer.tokenize(text.lower())

        if len(tokens) >= n:
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

            if len(ngrams) > 0:
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)
                distinct_score = unique_ngrams / total_ngrams
                distinct_scores.append(distinct_score)
            else:
                distinct_scores.append(0.0)
        else:
            distinct_scores.append(0.0)

    if len(distinct_scores) == 0:
        return 0.0

    avg_distinct_n = np.mean(distinct_scores)
    return avg_distinct_n


def calculate_distinct_1_2(texts):
    distinct_1 = calculate_distinct_n(texts, n=1)
    distinct_2 = calculate_distinct_n(texts, n=2)

    return distinct_1, distinct_2
