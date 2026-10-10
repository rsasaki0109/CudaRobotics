import unittest
from benchmark_kiss_icp_spatial import summarize


def rows(times):
    return [dict(stamp_ns=str(i * 100_000_000), **{field: str(t) for field in
            ("frame_ms", "odometry_ms", "normal_ms", "index_ms", "nn_ms")})
            for i, t in enumerate(times)]


class QueueMetrics(unittest.TestCase):
    def test_no_backlog_with_headroom(self):
        result = summarize(rows([50, 60, 50, 60]))
        self.assertEqual(result["fifo_final_response_ms"], 60)
        self.assertEqual(result["fifo_deadline_miss_fraction"], 0)

    def test_queue_accumulates_even_without_frame_drops(self):
        result = summarize(rows([150, 150, 150, 150]))
        self.assertEqual(result["fifo_final_response_ms"], 300)
        self.assertEqual(result["fifo_deadline_miss_fraction"], 1)

    def test_queue_catches_up_after_a_slow_frame(self):
        result = summarize(rows([250, 10, 10]))
        self.assertEqual(result["fifo_final_response_ms"], 70)
        self.assertAlmostEqual(result["fifo_deadline_miss_fraction"], 2 / 3)

    def test_nonfinite_timing_rejected(self):
        with self.assertRaises(ValueError):
            summarize(rows([float("nan"), 10]))


if __name__ == "__main__":
    unittest.main()
