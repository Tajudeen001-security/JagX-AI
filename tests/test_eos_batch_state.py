from __future__ import annotations

import pytest
import torch

from media.eos_batch_state import freeze_finished


def test_freeze_finished_preserves_active_rows_and_freezes_completed_rows():
    token = torch.tensor([[4], [7], [9]])
    finished = torch.tensor([False, True, True])
    assert torch.equal(freeze_finished(token, finished, 2), torch.tensor([[4], [2], [2]]))


def test_freeze_finished_validates_shapes_and_eos():
    with pytest.raises(ValueError, match="shape"):
        freeze_finished(torch.ones(3), torch.zeros(3, dtype=torch.bool), 2)
    with pytest.raises(ValueError, match="shape"):
        freeze_finished(torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.bool), 2)
    with pytest.raises(ValueError, match="non-negative"):
        freeze_finished(torch.ones(3, 1, dtype=torch.long), torch.zeros(3, dtype=torch.bool), -1)
