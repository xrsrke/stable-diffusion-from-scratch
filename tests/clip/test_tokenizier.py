from foundation.clip.tokenizier import CLIPTokenizier

def test_clip_tokenizer(input_ids):
    assert input_ids.shape == (1, 77)

def test_text_encoder(prompt_embedding):
    assert prompt_embedding.shape == (1, 77, 768)