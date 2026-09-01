import torch
from media.multimodal_contract import MediaTensor, Modality, MultimodalBatch
from multimodal.encoders import (
    AudioFrameEncoder,
    ImagePatchEncoder,
    MultimodalProjector,
    TextEncoder,
    VideoFrameEncoder,
)


def test_media_tensor_validate():
    t = MediaTensor(modality=Modality.TEXT, values=torch.zeros(1, 2, 8)).validate()
    assert t.modality is Modality.TEXT
    batch = MultimodalBatch(items=(t,)).validate()
    assert Modality.TEXT in batch.modalities()


def test_encoders_shapes():
    d_model = 32
    text = TextEncoder(d_model)
    img = ImagePatchEncoder(d_model, patch_dim=16)
    aud = AudioFrameEncoder(d_model, frame_dim=16)
    vid = VideoFrameEncoder(d_model, frame_dim=16)

    bt = MediaTensor(modality=Modality.TEXT, values=torch.randn(2, 5, d_model))
    bi = MediaTensor(modality=Modality.IMAGE, values=torch.randn(2, 7, 16))
    ba = MediaTensor(modality=Modality.AUDIO, values=torch.randn(2, 9, 16))
    bv = MediaTensor(modality=Modality.VIDEO, values=torch.randn(2, 4, 16))

    assert text(bt).shape == (2, 5, d_model)
    assert img(bi).shape == (2, 7, d_model)
    assert aud(ba).shape == (2, 9, d_model)
    assert vid(bv).shape == (2, 4, d_model)

    proj = MultimodalProjector(d_model, d_model)
    assert proj(img(bi)).shape == (2, 7, d_model)
