import datasets
import evaluate

from .bleu_ import (
    compute_bleu,
)
from .tokenizer_13a import Tokenizer13a


class Bleu(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(
                            datasets.Value("string", id="sequence"), id="references"
                        ),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(
        self,
        predictions,
        references,
        tokenizer=Tokenizer13a(),
        max_order=4,
        smooth=False,
    ):
        # if only one reference is provided make sure we still use list of lists
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        references = [[tokenizer(r) for r in ref] for ref in references]
        predictions = [tokenizer(p) for p in predictions]
        score = compute_bleu(
            reference_corpus=references,
            translation_corpus=predictions,
            max_order=max_order,
            smooth=smooth,
        )
        (bleu, precisions, bp, ratio, translation_length, reference_length) = score
        return {
            "bleu": bleu,
            "precisions": precisions,
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }
