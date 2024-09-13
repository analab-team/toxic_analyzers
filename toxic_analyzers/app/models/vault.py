from pydantic import BaseModel


class Vault(BaseModel):
    toxicity_threshold_output: float # 0.8
    toxicity_threshold_input: float # 0.8
    attention_threshold_percentile: float # 0.85
    top_k_tokens: int # 3
    