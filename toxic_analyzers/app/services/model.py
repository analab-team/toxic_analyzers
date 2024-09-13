from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.vault import Vault
from schemas.model_result import ModelResult, Reason

MODEL_CHECKPOINT = "cointegrated/rubert-tiny-toxicity"
WORD_DELIMITERS = (" ", "\n", "\t", ".", ",", "!", "?", ";", ":", "-")


class ToxicityModel:
    def __init__(self):
        self.analyzer = ToxicityAnalyzer()
        self.interpreter = AttentionInterpreter(self.analyzer.model, self.analyzer.tokenizer)

    def input_score(self, text: str, vault: Vault) -> ModelResult:
        """
        Проверяет текст на токсичность и возвращает оценку с индексами важных частей текста.
        """
        toxicity_score = self.analyzer.get_toxicity(text)
        important_spans = self.interpreter.analyze_toxicity(
            text, vault.attention_threshold_percentile, vault.top_k_tokens
        )

        reject_flg = toxicity_score > vault.toxicity_threshold_input
        reasons = [Reason(start=span[0], stop=span[1], additional_metric=toxicity_score) for span in important_spans]

        return ModelResult(metric=toxicity_score, reasons=reasons, reject_flg=reject_flg)

    def output_score(self, text: str, vault: Vault) -> ModelResult:
        """
        Аналогичная логика для выхода.
        """
        toxicity_score = self.analyzer.get_toxicity(text)
        important_spans = self.interpreter.analyze_toxicity(
            text, vault.attention_threshold_percentile, vault.top_k_tokens
        )
        
        reject_flg = toxicity_score > vault.toxicity_threshold_output
        reasons = [Reason(start=span[0], stop=span[1], additional_metric=toxicity_score) for span in important_spans]

        return ModelResult(metric=toxicity_score, reasons=reasons, reject_flg=reject_flg)


class ToxicityAnalyzer:
    """Класс для вычисления токсичности текста."""

    def __init__(self, model_checkpoint: str = MODEL_CHECKPOINT):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_toxicity(self, text: str) -> float:
        """
        Оценивает токсичность текста.

        Args:
            text (str): Текст для анализа.

        Returns:
            float: Суммарная токсичность текста.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = torch.sigmoid(outputs.logits).cpu().numpy()

        # Агрегируем массив в одно значение и возвращаем
        toxicity_score = 1 - proba.T[0] * (1 - proba.T[-1])
        return toxicity_score.item()


class AttentionInterpreter:
    """Класс для интерпретации внимания (attention) модели."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze_toxicity(self, text: str, attention_threshold_percentile: float, top_k: int) -> List[Tuple[int, int]]:
        """
        Интерпретирует внимание и возвращает индексы важных частей текста.

        Args:
            text (str): Входной текст.
            top_k (int): Количество ключевых токенов для возврата.

        Returns:
            List[Tuple[int, int]]: Важные части текста с их индексами [старт, конец].
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions[-1]  # Последний слой внимания
            avg_attentions = attentions.mean(dim=1)  # Усреднение по головам
            cls_attentions = avg_attentions[:, 0, :]  # Внимание от [CLS] токена

        # Убираем спец-токены ([CLS], [SEP])
        special_tokens_mask = offset_mapping[:, 0] == 0
        token_offsets = offset_mapping[~special_tokens_mask]
        token_attentions = cls_attentions[0, ~special_tokens_mask]

        # Нормализуем веса внимания
        token_attentions = token_attentions / token_attentions.sum()

        # Порог: выбираем токены по процентилю
        threshold = torch.quantile(token_attentions, attention_threshold_percentile)
        important_tokens_mask = token_attentions >= threshold
        important_offsets = token_offsets[important_tokens_mask]

        # Преобразуем в индексы [старт, конец] в тексте и расширяем до полных слов
        important_spans = []
        for offset in important_offsets:
            start, end = offset.tolist()
            start, end = self._expand_to_full_word(text, start, end)
            important_spans.append((start, end))

        return self._remove_overlapping_spans(important_spans)

    def _expand_to_full_word(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """
        Расширяет индексы токенов до границ полных слов.

        Args:
            text (str): Исходный текст.
            start (int): Начальный индекс.
            end (int): Конечный индекс.

        Returns:
            Tuple[int, int]: Расширенные индексы начала и конца слова.
        """
        while start > 0 and text[start - 1] not in WORD_DELIMITERS:
            start -= 1
        while end < len(text) and text[end] not in WORD_DELIMITERS:
            end += 1
        return start, end

    def _remove_overlapping_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Удаляет пересекающиеся интервалы индексов.

        Args:
            spans (List[Tuple[int, int]]): Список индексов [старт, конец].

        Returns:
            List[Tuple[int, int]]: Очищенный список без пересекающихся интервалов.
        """
        unique_spans = []
        for span in spans:
            if not any(s[0] <= span[1] and span[0] <= s[1] for s in unique_spans):
                unique_spans.append(span)
        return unique_spans
