import logging
import string
import threading
from queue import Queue

import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from helper import clean_text

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"

# summarization model
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
tokenizer_t5 = AutoTokenizer.from_pretrained("google-t5/t5-small")
summarizer_t5 = pipeline("summarization", model=model_t5, tokenizer=tokenizer_t5,
                         device=0 if device == "cuda" else -1)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def smart_chunking(text, max_chunk_size=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def text_summarizer(text, min_length=30, max_length=None):
    chunks = smart_chunking(text)
    summarized_chunks = []

    summary_queue = Queue()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def summarize_chunk(chunk):
        try:
            input_length = len(chunk.split())
            adjusted_max_length = max(30, int(0.5 * input_length))

            summary = summarizer_t5(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
            summary_text = summary[0]["summary_text"]
            logging.info("summary_text: %s", summary_text)
            summary_queue.put(summary_text)
        except Exception as e:
            logging.error(f"Summarization error: {e}. Using fallback model.")
            fallback_summary = call_gpt2_fallback("summarize the context:", chunk)
            summary_queue.put(fallback_summary)

    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=summarize_chunk, args=(chunk,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    while not summary_queue.empty():
        summarized_chunks.append(summary_queue.get())

    return " ".join(summarized_chunks)


def gpt2_generate_text(prompt, max_length=1024, temperature=1.0, top_p=0.9):
    try:
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")

        outputs = gpt2_model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id,
        )

        return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error generating text with GPT-2: {e}")
        return "GPT-2: Unable to generate a response."


def call_gpt2_fallback(query, relevant_context):
    try:
        gpt2_prompt = (
            f"Context:\n{relevant_context}\n\n"
            f"Answer the following question based on the context:\n"
            f"Question: {query}\n\nAnswer:"
        )
        gpt2_response = gpt2_generate_text(gpt2_prompt)
        return gpt2_response.strip() if gpt2_response else "I'm sorry, I couldn't generate a response for that."
    except Exception as e:
        logging.error(f"GPT-2 fallback error: {e}")
        return "I'm sorry, I couldn't generate a response for that."


def summarize_text(text):
    logging.info("input_text: %s", text)
    cleaned_text = preprocess_text(text)
    logging.info("cleaned_text: %s", cleaned_text)
    summary = text_summarizer(cleaned_text)
    logging.info("summary: %s", summary)
    cleansed_summary = clean_text(summary)
    logging.info("cleansed_summary: %s", cleansed_summary)
    return cleansed_summary
