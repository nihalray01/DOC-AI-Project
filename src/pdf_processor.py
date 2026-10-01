import re
from io import BytesIO
from typing import Any

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Remove unnecessary spaces and blank lines from extracted PDF text.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_from_pdf(uploaded_file: Any) -> list[dict]:
    """
    Extract readable text from every page of an uploaded PDF.

    Returns:
        A list containing page text, page number and file name.
    """
    extracted_pages: list[dict] = []

    try:
        file_bytes = uploaded_file.getvalue()
        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = clean_text(raw_text)

            if cleaned_text:
                extracted_pages.append(
                    {
                        "file_name": uploaded_file.name,
                        "page_number": page_number,
                        "text": cleaned_text,
                    }
                )

    except Exception as error:
        raise ValueError(
            f"Could not read '{uploaded_file.name}': {error}"
        ) from error

    return extracted_pages


def split_text_into_chunks(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Divide page text into overlapping chunks.

    Overlap helps preserve context between two consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks: list[dict] = []
    step_size = chunk_size - chunk_overlap

    for page in pages:
        text = page["text"]
        start_index = 0
        chunk_number = 1

        while start_index < len(text):
            end_index = start_index + chunk_size
            chunk_text = text[start_index:end_index].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "chunk_number": chunk_number,
                        "file_name": page["file_name"],
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    }
                )

            start_index += step_size
            chunk_number += 1

    return chunks


def process_uploaded_pdfs(
    uploaded_files: list[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[dict], list[str]]:
    """
    Extract and chunk text from multiple uploaded PDFs.

    Returns:
        chunks: Processed document chunks.
        errors: Errors encountered while reading individual files.
    """
    all_pages: list[dict] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            pages = extract_text_from_pdf(uploaded_file)
            all_pages.extend(pages)

            if not pages:
                errors.append(
                    f"No readable text found in '{uploaded_file.name}'. "
                    "It may be a scanned or image-only PDF."
                )

        except ValueError as error:
            errors.append(str(error))

    chunks = split_text_into_chunks(
        pages=all_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunks, errors