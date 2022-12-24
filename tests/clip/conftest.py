import pytest

from foundation.clip.tokenizier import CLIPTokenizier
from foundation.clip.text_encoder import TextEncoder

PROMPT = "persistence is all you need"

@pytest.fixture
def tokenized_prompt():
    tokenizier = CLIPTokenizier()
    output = tokenizier.encode(PROMPT)
    return output

@pytest.fixture
def input_ids(tokenized_prompt):
    return tokenized_prompt['input_ids']

@pytest.fixture
def prompt_ids(tokenized_prompt):
    return tokenized_prompt['input_ids']

@pytest.fixture
def prompt_embedding(prompt_ids):
    return TextEncoder.encode_from_ids(prompt_ids)