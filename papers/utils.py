import re

from pypdf import PdfReader


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MODEL_TOKEN_LIMIT = 1024
CHUNK_TOKEN_LIMIT = max(256, MODEL_TOKEN_LIMIT - 100)

_tokenizer = None
_summarizer = None


def _get_tokenizer():
    global _tokenizer, MODEL_TOKEN_LIMIT, CHUNK_TOKEN_LIMIT
    if _tokenizer is not None:
        return _tokenizer

    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        MODEL_TOKEN_LIMIT = _tokenizer.model_max_length if _tokenizer.model_max_length else MODEL_TOKEN_LIMIT
        CHUNK_TOKEN_LIMIT = max(256, MODEL_TOKEN_LIMIT - 100)
    except Exception:
        _tokenizer = None

    return _tokenizer


def _load_summarizer():
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    try:
        from transformers import pipeline
        _summarizer = pipeline(
            "summarization",
            model=MODEL_NAME
        )
        return _summarizer
    except Exception as exc:
        raise RuntimeError(
            "The Hugging Face summarization pipeline is unavailable. "
            "Install PyTorch, TensorFlow 2.x, or Flax to enable model inference. "
            f"Details: {exc}"
        )


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text if text.strip() else "No readable text found in PDF"

    except Exception as e:
        return f"PDF Error: {str(e)}"


def _split_text_into_chunks(text, max_tokens=CHUNK_TOKEN_LIMIT):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    tokenizer = _get_tokenizer()

    def token_count(value):
        if tokenizer is not None:
            tokens = tokenizer(value, return_tensors='pt', truncation=False)['input_ids'][0]
            return len(tokens)
        return len(value.split())

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if token_count(candidate) <= max_tokens:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if token_count(sentence) <= max_tokens:
            current_chunk = sentence
            continue

        words = sentence.split()
        word_chunk = ""
        for word in words:
            candidate = f"{word_chunk} {word}".strip() if word_chunk else word
            if token_count(candidate) <= max_tokens:
                word_chunk = candidate
            else:
                if word_chunk:
                    chunks.append(word_chunk)
                word_chunk = word

        if word_chunk:
            current_chunk = word_chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _heuristic_summary(text, summary_size='medium'):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return "No readable text found in PDF"

    if summary_size == 'small':
        count = min(3, len(sentences))
    elif summary_size == 'medium':
        count = min(6, len(sentences))
    else:
        count = min(10, len(sentences))

    return " ".join(sentences[:count])


def _summarize_chunks(chunks, max_len, min_len):
    summarizer = _load_summarizer()
    partial_summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        partial_summaries.append(result[0]['summary_text'])

    combined_summary = " ".join(partial_summaries)
    tokenizer = _get_tokenizer()
    if len(partial_summaries) == 1:
        return combined_summary

    if tokenizer is not None:
        count = len(tokenizer(combined_summary, return_tensors='pt', truncation=False)['input_ids'][0])
    else:
        count = len(combined_summary.split())

    if count <= CHUNK_TOKEN_LIMIT:
        result = summarizer(
            combined_summary,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return result[0]['summary_text']

    return combined_summary


def generate_summary(text, summary_size='medium'):
    if summary_size == 'small':
        max_len = 100
        min_len = 40
    elif summary_size == 'medium':
        max_len = 250
        min_len = 120
    else:  # large
        max_len = 500
        min_len = 400

    try:
        chunks = _split_text_into_chunks(text)
        if not chunks:
            return "No readable text found in PDF"

        try:
            summary_text = _summarize_chunks(chunks, max_len, min_len)
            return summary_text
        except RuntimeError:
            return _heuristic_summary(text, summary_size)
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

