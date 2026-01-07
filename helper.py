import os
import re
from collections import OrderedDict
from tempfile import NamedTemporaryFile
from typing import List, Generator, Union

from PyPDF2 import PdfReader
from docx import Document
from fastapi import UploadFile


class LRUCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove least used

        self.cache[key] = value


async def parse_file(
        file: UploadFile,
        chunk_size: int = 1000,
        large_file_threshold: int = 50
) -> Union[List[str], Generator[str, None, None]]:
    if file.filename.endswith(".pdf"):
        return parse_pdf(file, chunk_size, large_file_threshold)
    elif file.filename.endswith(".docx"):
        return parse_docx(file, chunk_size, large_file_threshold)
    elif file.filename.endswith(".txt"):
        return parse_txt(file, chunk_size, large_file_threshold)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")


def parse_pdf(file: UploadFile, chunk_size: int, large_pdf_threshold: int) -> Union[
    List[str], Generator[str, None, None]]:
    reader = PdfReader(file.file)
    total_pages = len(reader.pages)
    if total_pages <= large_pdf_threshold:
        content = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return chunk_text(content, chunk_size)
    else:
        return (page.extract_text() for page in reader.pages if page.extract_text())


def parse_docx(file: UploadFile, chunk_size: int, large_file_threshold: int) -> Union[
    List[str], Generator[str, None, None]]:
    with NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.file.read())
        temp_file_path = tmp.name

    try:
        document = Document(temp_file_path)
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        total_paragraphs = len(paragraphs)
        if total_paragraphs <= large_file_threshold:
            content = " ".join(paragraphs)
            return chunk_text(content, chunk_size)
        else:
            return (paragraph for paragraph in paragraphs)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def parse_txt(file: UploadFile, chunk_size: int, large_file_threshold: int) -> Union[
    List[str], Generator[str, None, None]]:
    lines = file.file.readlines()
    total_lines = len(lines)
    if total_lines <= large_file_threshold:
        content = " ".join(line.decode("utf-8").strip() for line in lines)
        return chunk_text(content, chunk_size)
    else:
        return (line.decode("utf-8").strip() for line in lines)


def chunk_text(content: str, chunk_size: int) -> List[str]:
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


async def get_file_size(file: UploadFile) -> float:
    current_position = file.file.tell()
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(current_position)

    return file_size / (1024 * 1024)


def clean_text(text: str) -> str:
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
