from __future__ import annotations

import pytest
import torch

from sightline.projector import Projector, calibrate_output_scale


def test_parameter_count_matches_formula() -> None:
    projector = Projector(input_width=4096, reader_width=2048)
    expected = 2 * 4096 + 4096 * 4096 + 4096 + 4096 * 2048 + 2048
    assert projector.parameter_count() == expected == 25_180_160


def test_parameter_count_k26_width() -> None:
    projector = Projector(input_width=4608, reader_width=2048)
    expected = 2 * 4608 + 4608 * 4608 + 4608 + 4608 * 2048 + 2048
    assert projector.parameter_count() == expected == 30_686_720


def test_output_width_equals_reader_hidden() -> None:
    projector = Projector(input_width=4096, reader_width=2048)
    out = projector(torch.randn(2, 5, 4096))
    assert out.shape == (2, 5, 2048)


def test_forward_accepts_4d_merged_features() -> None:
    projector = Projector(input_width=4096, reader_width=2048)
    out = projector(torch.randn(2, 5, 4, 1024))
    assert out.shape == (2, 5, 2048)


def test_hidden_width_override() -> None:
    projector = Projector(input_width=4096, reader_width=2048, hidden_width=8192)
    expected = 2 * 4096 + 4096 * 8192 + 8192 + 8192 * 2048 + 2048
    assert projector.parameter_count() == expected


def test_random_init_is_the_default() -> None:
    projector = Projector(input_width=4096, reader_width=2048)
    assert projector.init == "random"


def test_embedding_alignment_init_is_a_later_ablation() -> None:
    with pytest.raises(NotImplementedError, match="embedding"):
        Projector(input_width=4096, reader_width=2048, init="embedding_alignment")


def test_norm_ratio_is_positive_and_finite() -> None:
    projector = Projector(input_width=4096, reader_width=2048)
    features = torch.randn(2, 5, 4096)
    embedding_weight = torch.randn(128, 2048)
    ratio = projector.norm_ratio(features, embedding_weight)
    assert ratio > 0
    assert torch.isfinite(ratio)


def test_calibrate_output_scale_matches_the_reference_rms() -> None:
    torch.manual_seed(0)
    projector = Projector(input_width=8, reader_width=8)
    reference = torch.randn(64, 8) * 0.02
    features = torch.randn(4, 3, 8) * 5.0

    scale = calibrate_output_scale(projector, reference, features)

    produced = projector(features)
    rms_out = produced.float().pow(2).mean().sqrt()
    rms_ref = reference.float().pow(2).mean().sqrt()
    assert scale < 1.0
    assert torch.allclose(rms_out, rms_ref, rtol=1e-4)


def test_output_scale_multiplies_the_output() -> None:
    projector = Projector(input_width=8, reader_width=8)
    features = torch.randn(2, 3, 8)
    baseline = projector(features)
    projector.output_scale.fill_(0.0)
    assert torch.count_nonzero(projector(features)) == 0
    assert baseline.abs().sum() > 0
