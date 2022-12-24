from foundation.unet.condition_model import UNet2DConditionModel

def test_unet_conditional_model():
    model = UNet2DConditionModel()

    assert model.in_channels == 4