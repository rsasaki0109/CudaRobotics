# Adoption Metrics

This document defines the activation and retention funnel for CudaRobotics.
It is a product-improvement contract, not a release-readiness or benchmark
claim. Performance evidence continues to use the existing reproducibility
and release attestations.

## Outcomes

- A new visitor can select the right entry path without understanding the
  repository layout.
- A first-time user produces a visible and machine-readable result within 15
  minutes.
- A successful user has an obvious, progressively harder next action.
- Measurement does not add telemetry to the Python or ROS runtime by default.

## Funnel

| Stage | Definition | Primary signal |
|---|---|---|
| Visit | A view of the project landing page or documentation | GitHub and documentation aggregate traffic |
| Start | Opening Colab, the Python install guide, or the Nav2 guide | Aggregate outbound-link events on the documentation site |
| Activate | Producing the documented result for the selected path | Local success artifact or an optional aggregate completion event |
| Continue | Completing a different recipe 7–30 days after activation | Optional aggregate recipe completion event |
| Integrate | Running with a user-provided costmap, dataset, rosbag, or Nav2 configuration | Recipe-specific success artifact |

Event collection is not implemented yet. The repository currently produces
the local Colab completion artifact but does not receive it. Until a disclosed,
opt-in aggregate collection path is added, report only the available acquisition
signals and manually sampled activation runs; do not present estimated starts
or continuation as observed usage.

The Colab activation artifact is
`cudarobotics_quickstart_result.json`. Activation requires both the MPPI goal
check and registration check to pass. The local Python activation artifact is
`build/onboarding/python/python_quickstart_result.json`; it requires both child
quickstarts to exit successfully. The ROS 2 onboarding artifact is
`build/cudanav_closed_loop/manifest.json` from the smoke profile, with its
bound `mission_summary.json`. Do not compare their completion rates until the
start signals for each path are collected consistently.

## Metrics

Calculate each metric as a cohort ratio rather than a lifetime total:

- **Start rate:** unique starts / unique landing-page visitors.
- **Activation rate:** successful first-run artifacts / starts.
- **Time to first success:** elapsed time from start to successful artifact;
  report median and p90.
- **7–30 day continuation rate:** activated users completing a different
  recipe in that window / activated users eligible for the full window.
- **Integration rate:** activated users completing a user-data or Nav2 recipe
  within 30 days / activated users eligible for the full window.
- **Failure mix:** preflight, build, import, CUDA runtime, algorithm check, and
  unknown failures as percentages of failed starts.

Strict user-level continuation cannot be derived from GitHub, Colab, or
package-download aggregates alone. It requires an explicitly opt-in anonymous
identifier. Without one, report recipe completions as a continuation proxy and
label it as such rather than calling it a user retention rate.

Initial directional targets, to be replaced after four weeks of baseline data:

| Metric | Initial target |
|---|---|
| Colab activation rate | >= 60% |
| Median Colab time to first success | <= 10 min |
| Python activation rate | >= 50% |
| 7–30 day continuation rate | >= 25% |
| Known failure classification | >= 90% |

## Privacy and Collection Rules

- Do not add network telemetry to the installed Python package, C++ library,
  ROS nodes, or generated artifacts by default.
- Prefer aggregate GitHub, documentation, package-download, and CI statistics.
- If the Colab or documentation site sends a completion event, disclose it
  beside the action, collect no source data or GPU UUID, and use a rotating
  anonymous identifier with a short retention period.
- Treat stars, clones, and package downloads as acquisition signals, not
  activation or retention.
- Publish metric definitions and count failed attempts; do not report only
  successful runs.

## Four-Week Baseline

1. Tag the three README entry links so aggregate starts can be separated.
2. Validate the Colab success artifact on a clean GPU runtime.
3. Add matching local JSON success artifacts to the Python and Nav2
   quickstarts.
4. Record weekly cohorts for four weeks without changing the definitions.
5. Review the largest failure category and the highest-drop-off transition;
   ship one focused improvement, then compare equal-length cohorts.

Changes to metric definitions must update this file and note the effective
date so pre-change and post-change cohorts are not silently combined.

Aggregate explicitly shared local artifacts into one cohort without adding
runtime telemetry:

```bash
python scripts/summarize_adoption_results.py \
  shared-results/*.json \
  --cohort 2026-W35 \
  --output-json build/adoption/2026-W35.json \
  --output-markdown build/adoption/2026-W35.md
```

The report separates initial activation from continuation recipe completions,
computes median/p90 duration and failure mix by surface, and labels recipe
completion as a proxy rather than a user retention rate.
