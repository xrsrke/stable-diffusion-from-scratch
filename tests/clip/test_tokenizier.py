from foundation.clip.tokenizier import ClipTokenizer, TOKEN_LENGTH

def test_clip_tokenizer():
    tokenizier = ClipTokenizer()
    prompt = "persistence is all you need"

    tokens = tokenizier.encode(prompt)

    assert len(tokens) == TOKEN_LENGTH