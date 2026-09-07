"""hate_speech_toolkit (hst): a compact, reusable pipeline for hate/extremist-discourse studies.

Stages (each module is runnable with ``python -m hst.<module> --config project.yaml``):

  ingest      raw platform exports -> one record table per platform
  annotation  annotation exports -> training data; inter-annotator agreement
  classifiers train RoBERTa (multi-annotator or single-label); score the record tables
  llm         prompt-based LLM labelling (vLLM), shard merge, validation
  analysis    daily series, spikes, event windows + ITS, networks, geography
"""

__version__ = "0.1.0"
