from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from capabilities.finance.paper_trading import Order, PaperAccount, PaperBroker
from evaluation.frontier_gate import BenchmarkResult, CapabilityGate
from games.adapters import adapter_for
from media.generative import AudioGenerator, ImageGenerator, VideoGenerator
from media.movie_pipeline import MoviePipeline, MovieSpec


def test_native_media_backbones_forward():
    image = ImageGenerator(hidden=32)
    x = torch.randn(2, 4, 8, 8)
    y = image(x, torch.tensor([0.1, 0.2]))
    assert y.shape == x.shape
    audio = AudioGenerator(vocab_size=32, hidden=32, layers=1)
    assert audio(torch.randint(0, 32, (2, 5))).shape == (2, 5, 32)
    video = VideoGenerator(hidden=32)
    v = torch.randn(1, 4, 3, 8, 8)
    assert video(v, torch.tensor([0.2])).shape == v.shape


def test_game_adapters_create_projects():
    with tempfile.TemporaryDirectory() as d:
        for engine in ("godot", "unity", "unreal"):
            project = adapter_for(engine, d).create("demo", {"README.txt": engine})
            assert project.root.exists()
            assert (project.root / "README.txt").read_text(encoding="utf-8") == engine


def test_paper_broker_enforces_risk():
    account = PaperAccount(cash=10_000)
    broker = PaperBroker(account)
    ok, _ = broker.submit(Order("JGX", 1, 100, "buy"), {"JGX": 100})
    assert ok
    assert account.positions["JGX"] == 1


def test_frontier_gate_requires_all_results():
    gate = CapabilityGate(("reasoning", "coding"))
    results = [BenchmarkResult("reasoning", 0.9, 0.8), BenchmarkResult("coding", 0.7, 0.8)]
    assert not gate.frontier_claim_allowed(results)


def test_movie_pipeline_shot_planning_and_resume():
    class Scenes:
        def create_plan(self, spec):
            return {"title": spec.title}

    class Frames:
        def generate(self, shot, spec):
            return [shot.index]

    class Video:
        def generate(self, shot, keyframes, spec):
            return {"shot": shot.index, "frames": keyframes}

    pipeline = MoviePipeline(Scenes(), Frames(), Video())
    spec = MovieSpec("test", duration_seconds=21, shot_seconds=10)
    assert len(pipeline.shots(spec)) == 3
    assert set(pipeline.render(spec, completed={0})) == {1, 2}
