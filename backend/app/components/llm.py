import torch
from pathlib import Path
from transformers import AutoTokenizer, Gemma3ForCausalLM
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from typing import Optional

LOCAL_GEMMA_PATH = Path.home() / "Downloads" / "medical_chatbot" / "gemma_models" / "gemma-3-1b-it"
#LOCAL_GEMMA_PATH = Path.home() / "Downloads" / "medical_chatbot" / "gemma_models" / "gemma-3-1b-it-cpu"


def load_llm(local_path: Path = LOCAL_GEMMA_PATH):
    """
    Load Gemma 3 LLM from a local path, CPU fallback included.
    Returns: tokenizer, model
    """
    try:
        if not local_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {local_path}")

        tokenizer = AutoTokenizer.from_pretrained(local_path)

        device = torch.device("cpu")
        model = Gemma3ForCausalLM.from_pretrained(
            local_path,
            device_map=None,
            dtype=torch.float32  # <-- updated from torch_dtype
        ).to(device).eval()

        print("✅ Gemma 3 LLM loaded on CPU successfully")
        return tokenizer, model

    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        return None, None


class GemmaLLM(LLM):
    """LangChain-compatible LLM wrapper for Gemma 3"""
    model: Optional[any] = None
    tokenizer: Optional[any] = None
    max_new_tokens: int = 1024
    temperature: float = 0.7   # <--- added

    @property
    def _llm_type(self) -> str:
        return "gemma3"

    def _call(self, prompt: str, stop=None) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("LLM model or tokenizer not initialized")

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,     # <--- sampling ON
                top_p=0.9           # <--- stable sampling
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate(self, prompts, stop=None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)




def get_gemma_llm():
    tokenizer, model = load_llm()
    if tokenizer is None or model is None:
        raise RuntimeError("Failed to load Gemma 3 LLM")

    # Properly pass tokenizer and model to GemmaLLM
    llm = GemmaLLM(model=model, tokenizer=tokenizer)
    return llm
