# hate_speech_toolkit

A compact, reusable code base for studies of hateful and extremist discourse on social media,
distilled from three completed projects (Mardi Gras, anti-women, white-supremacist discourse).
It covers the whole path from raw platform exports to publishable tables and figures:

1. **ingest** – raw X (Brandwatch), Telegram and Instagram exports → one clean record table per platform
2. **annotation** – annotation-tool exports → training data, plus inter-annotator agreement
3. **classifiers** – fine-tune RoBERTa (multi-annotator model recommended) and score the record tables
4. **llm** – prompt-based labelling with an open LLM (vLLM), shard merging, validation against human labels
5. **analysis** – daily series, spike detection, event windows with interrupted time series, interaction networks, geography

Everything is driven by one YAML file per project (`configs/project_template.yaml`), and every
stage is a module you run with `python -m hst.<stage> --config my_project.yaml`.

## Install

```
conda env create -f environment.yml
conda activate hst
```

The LLM stage needs a GPU machine with vLLM (`pip install vllm`); everything else runs on a laptop.

## Running the pipeline

With a project config (see the next section) the stages run in this order:

```
python -m hst.ingest --config my_project.yaml                       # record tables
python -m hst.annotation.prepare_training_data --config my_project.yaml
python -m hst.annotation.agreement --config my_project.yaml
python -m hst.classifiers.train_multiannotator --config my_project.yaml --category <category>
python -m hst.classifiers.score --config my_project.yaml
python -m hst.llm.prepare_inputs --config my_project.yaml
python -m hst.llm.classify_vllm --config my_project.yaml            # needs a GPU with vLLM
python -m hst.llm.merge_shards --config my_project.yaml --attach
python -m hst.llm.validation --config my_project.yaml --reference <human_labels.csv>
python -m hst.analysis --config my_project.yaml
```

Every module accepts `--help`. Outputs land under the project's `paths.work` folder:
`records/`, `training/`, `llm/`, `analysis/` (tables) and `figures/`.

## Starting a new project

1. Copy `configs/project_template.yaml`, set the platforms, study period, categories
   (name, `target_group` as it appears in the annotations, threshold) and events.
2. Put raw exports under `paths.raw`: `raw/x/<export folders>` (Brandwatch bulk downloads),
   `raw/telegram/telegram_posts.csv` + `telegram_comments.csv`, `raw/instagram/instagram_posts.csv`
   + `instagram_comments.csv`.
3. Put annotation exports (columns `annotator, text, hate_presence, target_group`, plus `id` if the tool exports one)
   where `annotations.files` points.
4. Run the stages in the order shown above. Each stage writes a small summary JSON next to its output.

## The record table

Every stage reads and writes `work/records/<platform>_records.csv.gz`; see `hst/schema.py` for the
column list. Ingest creates it, scoring appends `<category>_probability` / `<category>_label` /
`any_hate`, the LLM merge appends `<category>_label` / `<category>_confidence` / `any_extremist`,
and the analysis modules only read it.

## Notes on the methods

* **Multi-annotator classifier** (`hst/classifiers/train_multiannotator.py`): a RoBERTa encoder with a
  consensus head plus one head per annotator, trained jointly. It uses every annotation, including
  disagreements, and exports a plain sequence-classification model. Use it when annotations record
  who labelled what; `train.py` is the plain single-label fallback.
* **Interrupted time series** (`hst/analysis/events.py`): a segmented negative-binomial regression on
  the daily series with trend, weekday and autocorrelation terms; reports the step change at the
  event with a confidence interval and draws the counterfactual.
* **Network layout** (`hst/analysis/network.py`): force-directed placement (DrL via python-igraph,
  Fruchterman–Reingold fallback) followed by a compaction step so communities read as distinct
  clusters, and comparable before/after snapshots drawn from one shared reference layout.

## Licence

MIT, see `LICENSE`.
