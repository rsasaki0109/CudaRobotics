#!/usr/bin/env python3
"""Regression tests for the CUDA acceleration static auditor."""

import tempfile
import unittest
from pathlib import Path

from audit_cuda_acceleration import inspect, kernel_bodies


SAMPLE = r"""
__global__ void serial_kernel(float *out, const float *in, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) out[i] += expf(in[j]);
  }
  atomicAdd(out, 1.0f);
}
void run(float *d, float *h, int bytes) {
  cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
}
"""


class AuditTest(unittest.TestCase):
    def test_kernel_extraction_balances_nested_braces(self):
        bodies = kernel_bodies(SAMPLE)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0].count("for ("), 2)

    def test_inspect_detects_acceleration_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.cu"
            path.write_text(SAMPLE, encoding="utf-8")
            row = inspect(path)
        self.assertEqual(row["kernels"], 1)
        self.assertEqual(row["kernel_loops"], 2)
        self.assertEqual(row["nested_kernel_loops"], 1)
        self.assertEqual(row["atomics"], 1)
        self.assertEqual(row["syncs"], 1)
        self.assertEqual(row["d2h"], 1)
        self.assertEqual(row["h2d"], 1)
        self.assertIn("nested kernel loops", row["signals"])


if __name__ == "__main__":
    unittest.main()
