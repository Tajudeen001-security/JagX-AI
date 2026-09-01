from evaluation.model_eval import coding_shape_smoke, instruction_following_smoke


def test_instruction_following_smoke():
    result = instruction_following_smoke()
    assert result.error is None
    assert result.passed
    assert result.samples == 3


def test_coding_shape_smoke():
    result = coding_shape_smoke()
    assert result.error is None
    assert result.passed
