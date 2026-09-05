# Speakrs DiariZen training

## Commercial four-corpus training

The full recipe uses AMI, AliMeeting, AISHELL-4, and the VoxConverse development
set for training. VoxConverse test remains held out. It also prepares the
three original test sets and VoxConverse test. The preparation programs keep
only mono 16 kHz FLAC files, so source archives do not consume disk space after
successful preparation.

The model architecture, random seed, chunk sizes, and learning rates match the
upstream WavLM Base+ recipe. A physical batch size of 16 and four gradient
accumulation steps provide the paper's effective batch size of 64 on a 16 GB
GPU. The run also records an epoch-zero baseline, all DER components, and the
five best model-only checkpoints. WavLM initialization is strict.

From the repository root:

```bash
python recipes/speakrs/export_wavlm_base_plus.py
cd recipes/speakrs
python prepare_full_corpus.py --plan
./run_full_pipeline.sh
```

For an unattended rental, start `supervise_full.sh` after the pipeline. It
monitors the current PID, retries resumable failures up to five times, runs the
held-out evaluation after a clean training completion, and writes
`<experiment>.status` as `ready_to_stop` only after every score succeeds. The
supervisor derives the experiment name from `DIARIZEN_TRAINING_CONFIG` and
rejects a conflicting `DIARIZEN_EXPERIMENT_ID`.
Completion must appear in output from the current monitored attempt. Old log
lines do not skip training; resuming a terminal checkpoint confirms completion
without another training epoch.

The full pipeline is resumable. If training is interrupted after data
preparation, use:

```bash
./run_full.sh --resume
```

The default configuration is
`conf/full_wavlm_base_plus_16gb_upstream_v2.toml`. It includes the corrected
gradient accumulation path and the current upstream-equivalent AISHELL audio
policy. Set `DIARIZEN_TRAINING_CONFIG` to start or resume a different
configuration. The configuration file name also defines the experiment name.

To select the default configuration explicitly, use:

```bash
DIARIZEN_TRAINING_CONFIG="$PWD/conf/full_wavlm_base_plus_16gb_upstream_v2.toml" \
    ./run_full.sh
```

Use the same environment variable with the unattended supervisor:

```bash
DIARIZEN_TRAINING_CONFIG="$PWD/conf/full_wavlm_base_plus_16gb_upstream_v2.toml" \
    ./supervise_full.sh
```

After training, `evaluate_full.sh` averages the five best validation-loss
checkpoints. It runs the upstream constrained AHC pipeline and dscore with a
zero-second collar on each held-out test set. The public pyannote WeSpeaker
checkpoint and the `nryant/dscore` repository must be present at the paths in
the script. Install `requirements-evaluation.txt` in the training environment.
Neither model download needs a secret token.

Each inference directory contains a run manifest. The evaluator reuses a
partial result only when its audio inputs, configuration, model checkpoints,
embedding, inference settings, installed engine source, and dependency versions
match. VBx runs also bind the PLDA asset contents. Engine-bound results use a
new output profile, so earlier results remain intact. Set `DIARIZEN_EXPERIMENT_ID` when
evaluating an experiment other than
`full_wavlm_base_plus_16gb_upstream_v2`.

Run `evaluate_official_control.sh` to check the complete evaluation path with
the published `diarizen-meeting-base` checkpoint before comparing a trained
checkpoint. The script requires the official `config.toml` and
`pytorch_model.bin` under `artifacts/diarizen-meeting-base`. Each corpus must
match its published collar-zero DER within one point.

AISHELL-4 uses an equal arithmetic mean of all eight microphones, which matches
the upstream SoX mono conversion. A versioned sidecar binds each prepared file
to this policy and prevents reuse of audio from an older downmix. AliMeeting
uses its first far-field channel because the model selects channel zero. AMI
uses Array1 microphone 1. `run_full.sh` verifies the complete preparation
identity and the exact 732-recording training manifest before it starts
training. After rebuilding the standard corpora, use
`prepare_voxconverse.py --reuse-prepared-audio` to restore and seal the
combined manifests without downloading the two audio archives again.

The training data excludes RAMC, MSDWild, and DIHARD-3 because their terms do
not permit commercial use. AMI and the VoxConverse data are CC BY 4.0.
AISHELL-4 and AliMeeting are CC BY-SA 4.0. The torchaudio WavLM Base+
initialization is MIT, and the WeSpeaker embedding checkpoint is CC BY 4.0.
Distributed model weights therefore require attribution and may require
ShareAlike terms because of the training data. VoxConverse copyright remains
with the original video owners.

The standard model has capacity for four active speakers per chunk. The source
manifests contain 130 of 172,339 training chunks and 11 of 10,848 development
chunks above that capacity. The upstream collation policy retains the four most
active speakers in these chunks. Run `audit_speaker_capacity.py` to reproduce
these counts.

## AMI-SDM pilot

This recipe trains the DiariZen WavLM Base+ and Conformer segmentation model on
a deterministic 14.4-hour subset of AMI single distant microphone audio. It
uses the public `torchaudio.pipelines.WAVLM_BASE_PLUS` checkpoint as the model
initialization and does not use the non-commercial DiariZen model weights.

The default configuration targets one NVIDIA GPU with 16 GB of memory. It uses
a physical batch size of 16 and four gradient accumulation steps to preserve
the paper's effective batch size of 64.

### Run

From the repository root:

```bash
python recipes/speakrs/export_wavlm_base_plus.py
python recipes/speakrs/prepare_ami_sdm.py
recipes/speakrs/run.sh
```

The preparation commands are resumable. Existing complete audio files are
verified and reused. The training command resumes from the newest checkpoint
when it is called with `--resume`.

```bash
recipes/speakrs/run.sh --resume
```

Data and model licenses must be reviewed before distribution. This recipe uses
AMI data under CC BY 4.0 and code from DiariZen and torchaudio. The WavLM Base+
model provenance and all generated file hashes are recorded in the recipe
artifacts.
