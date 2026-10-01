import os

from dotenv import load_dotenv
from groq import Groq

from src.prompts import build_rag_prompt


load_dotenv()


class RAGEngine:

    def __init__(self) -> None:
        self.api_key = os.getenv(
            "VITE_GROQ_API_KEY",
            "",
        ).strip()

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        ).strip()

        self.client = None

        if self.api_key:
            self.client = Groq(
                api_key=self.api_key
            )

    def is_configured(self) -> bool:
        return bool(
            self.api_key
            and self.api_key != "PASTE_YOUR_API_KEY_HERE"
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:

        if not question.strip():
            return "Please enter a valid question."

        if not retrieved_chunks:
            return (
                "I could not find this information "
                "in the uploaded documents."
            )

        if not self.is_configured():
            return self._create_fallback_answer(
                retrieved_chunks
            )

        prompt = build_rag_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful document assistant. "
                            "Answer only using the provided document context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )

            answer = response.choices[0].message.content

            if not answer:
                return self._create_fallback_answer(
                    retrieved_chunks
                )

            return answer.strip()

        except Exception as error:
            return (
                "Groq API could not generate the answer.\n\n"
                f"Technical detail: {error}"
            )

    @staticmethod
    def _create_fallback_answer(
        retrieved_chunks: list[dict],
    ) -> str:

        best_chunk = retrieved_chunks[0]

        return (
            "The most relevant document section is:\n\n"
            f"{best_chunk['text']}\n\n"
            "### Source\n"
            f"- {best_chunk['file_name']}, "
            f"Page {best_chunk['page_number']}"
        )