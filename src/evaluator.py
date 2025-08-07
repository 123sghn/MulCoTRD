import re
from typing import List

PREDICTION_PREFIXES = {None: None, "ft_cot_token": "Prediction"}


class Evaluator:
    def __init__(self, dataset_key, task_type="ft_cot_token"):
        self.dataset_key = dataset_key
        self.prediction_prefix = PREDICTION_PREFIXES[task_type]

    def _extract_prediction_candidates(self, prediction: str) -> List[str]:
        """
        Extracts all potential answer predictions which satisfy the dataset's answer format from the
        prediction string
        """

        original_prediction = [prediction]

        if self.dataset_key in ("SAS", "SAM", "T15", "T17"):
            prediction = prediction.lower()
            prediction = re.sub("\"|'|\n|\.|\s|\:|\,", " ", prediction)
            prediction = prediction.split(" ")
            prediction = [
                i for i in prediction if i in ("positive", "neutral", "negative")
            ]
        else:
            raise ValueError("Invalid dataset: {}".format(self.dataset_key))

        if len(prediction) != 0:
            return prediction
        else:
            return original_prediction

    def cleanse_prediction(self, completion, return_all=False):
        if self.prediction_prefix is None:
            # If no prefix, use first candidate
            predictions = self._extract_prediction_candidates(completion)
            first = True
        else:
            index = completion.find(self.prediction_prefix)
            if index == -1:
                # If prefix not found, use *last* candidate
                predictions = self._extract_prediction_candidates(completion)
                first = False
            else:
                # If prefix found, use *first* candidate after prefix
                start_of_answer = index + len(self.prediction_prefix)
                predictions = self._extract_prediction_candidates(
                    completion[start_of_answer:]
                )
                first = True

        answer = None
        if predictions:
            answer = predictions[0] if first else predictions[-1]

        return (answer, predictions) if return_all else answer

    def cleanse_answer(self, answer: str) -> str:
        if self.dataset_key in ("SAS", "SAM", "T15", "T17"):
            answer = answer.lower()
        return answer

    def _compare_prediction_and_answer(self, prediction, answer) -> bool:
        return prediction is not None and prediction == answer

    def evaluate_single_instance(self, prediction, answer) -> bool:
        cleanse_prediction = self.cleanse_prediction(prediction)
        cleanse_answer = self.cleanse_answer(answer)
        evaluation = self._compare_prediction_and_answer(
            cleanse_prediction, cleanse_answer
        )
        return evaluation
