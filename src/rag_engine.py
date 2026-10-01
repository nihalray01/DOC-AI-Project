import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """
    Generate grounded, context-aware answers using Groq or Gemini API and retrieved PDF context.
    """

    def __init__(self) -> None:
        self.provider: str = "none"
        self.model_name: str = ""
        self.groq_client: Optional[Any] = None
        self.gemini_client: Optional[Any] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        Auto-detect available API keys and initialize the appropriate client.
        Prioritizes Groq API, then Google Gemini API.
        """
        # Check Groq API Key
        groq_key = (
            os.getenv("GROQ_API_KEY", "")
            or os.getenv("VITE_GROQ_API_KEY", "")
        ).strip()

        if groq_key and groq_key != "PASTE_YOUR_API_KEY_HERE":
            try:
                from groq import Groq

                self.groq_client = Groq(api_key=groq_key)
                self.provider = "groq"
                self.model_name = os.getenv(
                    "GROQ_MODEL", "openai/gpt-oss-120b"
                ).strip()
                return
            except Exception as error:
                print(f"Failed to initialize Groq client: {error}")

        # Check Gemini API Key
        gemini_key = (
            os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        ).strip()

        if gemini_key and gemini_key != "PASTE_YOUR_API_KEY_HERE":
            try:
                from google import genai

                self.gemini_client = genai.Client(api_key=gemini_key)
                self.provider = "gemini"
                self.model_name = os.getenv(
                    "GEMINI_MODEL", "gemini-2.5-flash"
                ).strip()
                return
            except Exception as error:
                print(f"Failed to initialize Gemini client: {error}")

    def is_configured(self) -> bool:
        """
        Check if a valid LLM API provider is configured.
        """
        return self.provider in ("groq", "gemini")

    def get_provider_info(self) -> Dict[str, str]:
        """
        Return active provider name, model, and configuration status.
        """
        if self.provider == "groq":
            return {
                "provider": "Groq AI",
                "model": self.model_name,
                "status": "Connected",
            }
        elif self.provider == "gemini":
            return {
                "provider": "Google Gemini",
                "model": self.model_name,
                "status": "Connected",
            }
        return {
            "provider": "Fallback (Retrieval Only)",
            "model": "None",
            "status": "Not Configured",
        }

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate grounded answer using retrieved document chunks and configured LLM.
        """
        if not question.strip():
            return "Please enter a valid question."

        if not retrieved_chunks:
            return (
                "I could not find relevant information "
                "in the uploaded documents to answer your question."
            )

        if not self.is_configured():
            return self._create_fallback_answer(retrieved_chunks)

        from src.prompts import build_rag_prompt

        prompt = build_rag_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        if self.provider == "groq" and self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are DocuMind AI, an expert document assistant. "
                                "Answer the user's question using ONLY the provided document context. "
                                "If the answer is not present in the context, state that clearly."
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
                if answer and answer.strip():
                    return answer.strip()

                return self._create_fallback_answer(retrieved_chunks)

            except Exception as error:
                return (
                    f"⚠️ **Groq API Error**: {error}\n\n"
                    + self._create_fallback_answer(retrieved_chunks)
                )

        elif self.provider == "gemini" and self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                if response.text and response.text.strip():
                    return response.text.strip()

                return self._create_fallback_answer(retrieved_chunks)

            except Exception as error:
                return (
                    f"⚠️ **Gemini API Error**: {error}\n\n"
                    + self._create_fallback_answer(retrieved_chunks)
                )

        return self._create_fallback_answer(retrieved_chunks)

    @staticmethod
    def _create_fallback_answer(
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Create a fallback answer when API key is missing or calls fail.
        """
        best_chunk = retrieved_chunks[0]

        return (
            "ℹ️ *LLM API is not configured or unavailable. "
            "Showing top matched section directly from document:*\n\n"
            f"> {best_chunk['text']}\n\n"
            f"**Source**: `{best_chunk['file_name']}` (Page {best_chunk['page_number']})"
        )