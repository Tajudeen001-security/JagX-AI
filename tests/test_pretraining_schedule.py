import torch

from training.pretraining import PretrainingConfig, build_optimizer, build_scheduler


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_warmup_cosine_schedule_has_expected_shape():
    model = Tiny()
    cfg = PretrainingConfig(max_steps=10, warmup_steps=2, learning_rate=1e-3, min_lr_ratio=0.1)
    opt = build_optimizer(model, cfg)
    scheduler = build_scheduler(opt, cfg)
    values = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        opt.step()
        scheduler.step()
        values.append(opt.param_groups[0]["lr"])
    assert values[0] > 0
    assert values[1] >= values[0]
    assert max(values) <= cfg.learning_rate
    assert values[-1] >= cfg.learning_rate * cfg.min_lr_ratio * 0.99


def test_invalid_schedule_is_rejected():
    try:
        PretrainingConfig(max_steps=10, warmup_steps=11).validate()
    except ValueError:
        return
    raise AssertionError("warmup longer than max_steps should be rejected")
