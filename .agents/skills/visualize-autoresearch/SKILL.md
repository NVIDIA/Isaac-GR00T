---
name: visualize-autoresearch
description: Render GR00T AutoResearch state.json files as self-contained interactive HTML reports. Use when Codex needs to visualize a study's summary, metric results, trial lineage or composition DAG, score improvements and regressions, stage failures, or per-trial details.
---

# Visualize AutoResearch

Generate one portable HTML file from durable AutoResearch state without modifying the study.

## Render a report

Run from the repository root:

    python3 skills/visualize-autoresearch/scripts/render.py \
      /path/to/state.json \
      --output /path/to/autoresearch-report.html

Omit `--output` to create `state.visualization.html` beside the input file.

The renderer uses only the Python standard library. The generated HTML has no CDN, package, font, or server dependency and can be opened directly in a browser.

## Interpret the diagram

- Treat the baseline as the root node.
- Group wave-based studies into one column per wave so parallel trials remain aligned. Mark active_wave_index as active and completed earlier waves as settled.
- Draw solid arrows from each parent to its trial.
- Draw dashed purple arrows for composition-source changes.
- Color scored trials green when better than their parent and red when worse. Interpret score direction from `state.direction`.
- Normalize color strength to the largest absolute parent-relative change in the study.
- Keep ties, pending, running, skipped, failed, and otherwise unscored trials visually distinct. Never label an unscored failure as a regression.
- Mark `best_trial_id` separately from parent-relative improvement.
- Open a trial's detail drawer to inspect its hypothesis, result summary, next guidance, score, code revision, changed files, train/eval attempts, errors, and checkpoint.

## Validate the output

After rendering:

1. Confirm the command prints the expected absolute HTML path.
2. Confirm the HTML file is non-empty.
3. For a new state schema, run the skill tests:

       .venv/bin/python -m pytest -q \
         tests/skills/autoresearch/test_visualize_autoresearch_skill.py

If the state directory is read-only or out of quota, pass a writable `--output` path. Do not alter `state.json` to make a report render.
