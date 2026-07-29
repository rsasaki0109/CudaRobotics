#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest

from render_contact_paper_results import render


class RenderContactPaperResultsTest(unittest.TestCase):
    def test_release_results_and_limitations_are_rendered(self):
        output = render()
        for expected in (
            "32,400",
            "0.800",
            "+0.533",
            "0.000305",
            "MuJoCo 3.11.0",
            "0.457",
            "six Holm-significant negative cells",
            "0/30 on `box_align_detour`",
            "not a standard manipulator benchmark or real-robot result",
        ):
            self.assertIn(expected, output)

    def test_output_has_no_absolute_workspace_path(self):
        output = render()
        self.assertNotIn(str(Path.cwd().resolve()), output)


if __name__ == "__main__":
    unittest.main()
