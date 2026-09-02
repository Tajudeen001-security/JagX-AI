import torch

from media.conditioning import IdentityCodec, ModalityProjector, ModalityRouter


def test_projector_shape():
    model = ModalityProjector(8, 16)
    assert model(torch.randn(2, 5, 8)).shape == (2, 5, 16)


def test_router_fuses_modalities():
    router = ModalityRouter(16, modalities=2)
    out = router(torch.randn(2, 3, 16), torch.randn(2, 4, 16), torch.randn(2, 2, 16))
    assert out.shape == (2, 9, 16)


def test_identity_codec():
    codec = IdentityCodec()
    x = torch.randn(2, 3)
    assert torch.equal(codec.decode(codec.encode(x)), x)
