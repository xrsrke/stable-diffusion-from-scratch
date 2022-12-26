from foundation.clip.tokenizier import CLIPTokenizer

def test_create_clip_tokenizier():
    tokenizer = CLIPTokenizer()

    assert tokenizer.model_max_length == 77

def test_clip_tokenizer(input_ids):
    assert input_ids.shape == (1, 77)

def test_text_encoder(prompt_embedding):
    assert prompt_embedding.shape == (1, 77, 768)