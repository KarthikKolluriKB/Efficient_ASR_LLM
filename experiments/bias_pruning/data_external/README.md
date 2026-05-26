# External fairness datasets — download guide

Five datasets to extend Experiment 1 beyond CommonVoice 22. None of them
require GPU; downloads can run in parallel while the GPU is busy.

All scripts default to dropping data under `data/<dataset_name>/` at the
project root. Override with `--output_dir` if you want it elsewhere
(useful when the project drive is small and you'd rather stage downloads
on a scratch disk).

---

## At a glance

| Dataset | Downloader | Auto? | Size on disk | License | Bias dimensions added |
|---|---|:---:|---:|---|---|
| [EdAcc](#edacc) | `download_edacc.py` | ✅ fully | ~6.95 GB | CC-BY-SA | accent (25), L1 (29), gender (3) |
| [L2-ARCTIC](#l2-arctic) | `download_l2arctic.py` | ✅ after HF login | ~473 MB | CC-BY-NC | L1 (6), native vs non-native |
| [Artie Bias](#artie-bias-corpus) | `download_artie.py` | ⚠️ TSV auto, **audio CDN dead** | ~5 MB TSV; audio you supply | CC-0 | gender, age, accent (balanced) |
| [CORAAL](#coraal) | `download_coraal.py` | ✅ direct URLs | tens of GB total | CC BY-NC-SA | regional AAL, age, sex, education |
| [Fair-Speech](#fair-speech) | `download_fairspeech.py` | ❌ no public URL surfaced | unknown | TBD | gender, age, ethnicity, geo, native flag |

---

## EdAcc

```bash
python experiments/bias_pruning/data_external/download_edacc.py
```

- Source: `edinburghcstr/edacc` on HF Hub.
- 40 h of dyadic English conversations, 25 standardised accent labels.
- 32 kHz audio — your pipeline expects 16 kHz; resample at HF-dataset-build
  time using the existing `datamodule/hf_data.py` pattern (`librosa.resample`).
- Highest-value addition for accent stratification.

## L2-ARCTIC

```bash
# One-time: visit https://huggingface.co/datasets/KoelLabs/L2Arctic, accept terms
huggingface-cli login                   # paste a token with gated-repo read
python experiments/bias_pruning/data_external/download_l2arctic.py
```

- Source: `KoelLabs/L2Arctic` on HF Hub (gated, free).
- 24 non-native speakers across Hindi / Korean / Mandarin / Spanish / Arabic / Vietnamese.
- Already at 16 kHz. Small (~473 MB) — good first non-CV experiment.

## Artie Bias Corpus

```bash
python experiments/bias_pruning/data_external/download_artie.py
```

**Status (May 2026):** the original Artie audio CDN
(`http://ml-corpora.artie.com/`) is dead — DNS no longer resolves, the
host returns `ECONNREFUSED`. Artie Inc. was acquired by Match Group and
the corpus was retired with no announced successor. No mirror found on
HuggingFace, Zenodo, or archive.org as of May 2026.

What the downloader can still do:
- Clone `artie-inc/artie-bias-corpus` for the TSV (same column schema
  as CommonVoice: `client_id, path, sentence, up_votes, down_votes,
  age, gender, accent`) and the original bias-detection scripts.

What still has to be done manually:
- Source the 1,712 MP3 clips from a copy of **CommonVoice English June 2019**
  (Mozilla Data Collective since Oct 2025 — or a local mirror if you,
  the lab, or collaborators kept one).
- Place each clip under `data/artie_bias/clips/<path>` matching the
  TSV's `path` column.

**Recommendation:** Artie has become the hardest of the five external
datasets to operationalise. Its scientific value (~2.4 h, balanced
cells, CC-0) is real but the audio-sourcing cost is now high enough
that EdAcc + Fair-Speech are a better starting pair. Skip Artie unless
you specifically need its balanced-by-design cell structure as a
confirmation.

## CORAAL

```bash
# Inspect what's available without downloading
python experiments/bias_pruning/data_external/download_coraal.py --list_only

# Try one component first (DCA is a reasonable starter)
python experiments/bias_pruning/data_external/download_coraal.py --components DCA

# Full corpus once you've confirmed the path
python experiments/bias_pruning/data_external/download_coraal.py --all
```

- Source: <http://lingtools.uoregon.edu/coraal/> (HTTP, not HTTPS; cert chain
  is occasionally broken — the script bypasses verification on purpose).
- The site is component-based (ATL, DCA, DCB, DTA, LES, PRV, ROC, VLD, ...);
  each component has its own audio + transcript archives.
- **Major gotcha:** audio files are 30–60 min interviews. The current
  inference pipeline expects ≤30 s segments. Before evaluation you must
  chunk audio at transcript timestamps from the accompanying TextGrid /
  EAF / TXT files. Plan ~3–5 days of data engineering before CORAAL is
  usable for the bias pipeline.

## Fair-Speech (Meta ASR Fairness Evaluation Dataset)

Meta releases this via a click-to-accept download page — two files
(`asr_fairness_audio.zip`, ~4.1 GB; `asr_fairness_metadata.tsv`).
There's no scriptable download; do the browser step once, then:

```bash
python experiments/bias_pruning/data_external/download_fairspeech.py \
    --zip_path ~/Downloads/asr_fairness_audio.zip \
    --metadata_path ~/Downloads/asr_fairness_metadata.tsv
```

The script stages both files under `data/fairspeech/`, extracts the
audio, and prints the license reminder below.

**License restrictions — important.** Meta's terms forbid redistribution
of any part of the dataset, including reference transcripts:

| Action | Allowed? |
|---|:---:|
| Use to evaluate ASR fairness (this experiment) | ✅ explicit Purpose |
| Keep raw audio / metadata under `data/` (gitignored) | ✅ stays local |
| Publish summary-level results (per-gender WER, p-values) | ✅ derivative IP we own |
| Commit per-utterance CSV containing this dataset's text | ❌ blocked by `.gitignore` pattern `*fairspeech*` |
| Train a model that predicts race / ethnicity / gender of individuals | ❌ forbidden |
| Redistribute audio / metadata to a third party | ❌ forbidden |
| Re-identify any individual | ❌ forbidden |

- 30,000 utterances, 602 US speakers, avg 7.36 s per clip (~62 h total)
- Demographics: age, gender, ethnicity, geographic location, native-English
- Paper: [arXiv 2408.12734](https://arxiv.org/abs/2408.12734)

---

## Recommended download order

Start the cheap, fully-automated ones in parallel; chase the manual steps
afterwards.

```bash
# Run these three in parallel (different terminals or with &)
python experiments/bias_pruning/data_external/download_edacc.py        &
python experiments/bias_pruning/data_external/download_l2arctic.py     &
python experiments/bias_pruning/data_external/download_coraal.py --components DCA &
wait

# Then the manual ones, in any order
python experiments/bias_pruning/data_external/download_artie.py
python experiments/bias_pruning/data_external/download_fairspeech.py   # after configuring the source
```

---

## What downloading does NOT solve

These scripts only place raw data on disk. For each new dataset you'll
also need:

1. **An HF-format builder** (mirroring `datamodule/hf_data.py`) — turns
   the raw audio + metadata into `{audio_array, transcription,
   speaker_id, duration}` so the existing inference loader works.
2. **A demographic adapter** — replaces `load_cv_demographics()` in
   `evaluate_subgroup_wer.py` with a function that reads this dataset's
   metadata schema.
3. **A text-normalisation decision** — especially for CORAAL, where
   dialectal spellings ("ain't", "gonna") *should not* be silently
   normalised away; that is itself a fairness choice.

The downloads can run unattended; the adapters are a second phase.
