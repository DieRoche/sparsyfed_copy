"""Unit tests for FLOP logging helpers in WandbServer."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from project.fed.server.wandb_server import WandbServer


class WandbServerFlopTests(unittest.TestCase):
    """Test suite covering FLOP accumulation helpers."""

    def setUp(self) -> None:  # noqa: D401 - short description inherited
        self.server = object.__new__(WandbServer)
        WandbServer._reset_flop_state(self.server)

    def test_aggregate_round_flops_from_results(self) -> None:
        """Round FLOPs are summed across fit results."""

        fit_results = [
            (None, SimpleNamespace(metrics={"round_flops": 5, "round_flops_compression": 2})),
            (None, SimpleNamespace(metrics={"round_flops": 3, "round_flops_decompression": 1})),
        ]
        fit_metrics: dict[str, float] = {}

        WandbServer._update_flop_metrics(self.server, fit_results, fit_metrics)

        self.assertTrue(self.server._flop_metrics_available)
        self.assertAlmostEqual(fit_metrics["round_flops"], 8.0)
        self.assertAlmostEqual(fit_metrics["round_flops_compression"], 2.0)
        self.assertAlmostEqual(fit_metrics["round_flops_decompression"], 1.0)
        self.assertAlmostEqual(
            fit_metrics["round_flops_including_compression"],
            11.0,
        )
        self.assertAlmostEqual(fit_metrics["total_flops"], 8.0)
        self.assertAlmostEqual(fit_metrics["total_flops_compression"], 2.0)
        self.assertAlmostEqual(fit_metrics["total_flops_decompression"], 1.0)
        self.assertAlmostEqual(
            fit_metrics["total_flops_including_compression"],
            11.0,
        )

    def test_totals_from_metrics_override_accumulation(self) -> None:
        """Totals provided by aggregated metrics are adopted directly."""

        fit_results: list[tuple[None, SimpleNamespace]] = []
        fit_metrics = {
            "total_flops": 100,
            "total_flops_compression": 10,
            "total_flops_decompression": 5,
            "total_flops_including_compression": 115,
        }

        WandbServer._update_flop_metrics(self.server, fit_results, fit_metrics)

        self.assertTrue(self.server._flop_metrics_available)
        self.assertAlmostEqual(self.server._flop_totals["total_flops"], 100.0)
        self.assertAlmostEqual(
            fit_metrics["round_flops_including_compression"],
            115.0,
        )
        self.assertAlmostEqual(fit_metrics["total_flops"], 100.0)
        self.assertAlmostEqual(
            fit_metrics["total_flops_including_compression"],
            115.0,
        )


if __name__ == "__main__":  # pragma: no cover - convenient direct execution
    unittest.main()
