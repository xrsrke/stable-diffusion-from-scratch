import pytest

from foundation.clip.tokenizier import CLIPTokenizer
from foundation.clip.text_encoder import CLIPTextEncoder

PROMPT = "persistence is all you need"

@pytest.fixture
def tokenized_prompt():
    tokenizier = CLIPTokenizer()
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
    return CLIPTextEncoder.encode_from_ids(prompt_ids)