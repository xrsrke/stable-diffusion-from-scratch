from foundation.clip.text_encoder import CLIPTextEncoder

def test_text_encoder(input_ids):
    output = CLIPTextEncoder.encode_from_ids(input_ids)

    assert output.shape == (1, 77, 768)