# Quick start

## Choose a project folder

Data, cache, results and model weights go into sub-folders of it:

```bash
neuroatlas config init ~/neuroatlas
```

To keep one of them elsewhere, add for example `--data-root /path/to/datasets`.
[Configuration](../guide/configuration.md) explains the four folders.

## Run sleep staging

Download Sleep-EDF and the BIOT weights, then run sleep staging:

```bash
neuroatlas data download sleep_edf_expanded --mirror aws
neuroatlas models download biot_pretrained

neuroatlas check sleep_stage -m biot_pretrained          # quick test on one batch
neuroatlas run sleep_stage -m biot_pretrained --debug    # fold 0 only
neuroatlas run sleep_stage -m biot_pretrained            # all folds
neuroatlas results sleep_stage
```

The [walkthrough](walkthrough.md) is a full example with the expected output.

## Run any benchmark

Every benchmark runs the same way:

```bash
neuroatlas show epilepsy                          # description, datasets and commands
neuroatlas run epilepsy -m cbramod_pretrained     # its default dataset (Siena)
neuroatlas run epilepsy -m all_fm --dataset full  # all EEG FMs on all datasets you have
neuroatlas results epilepsy
```

`neuroatlas list benchmarks` lists all twelve, and `neuroatlas show <benchmark>` describes one.
[Benchmarks](../guide/benchmarks.md) describes them by domain.

To run many jobs on a cluster, `neuroatlas submit` writes one HTCondor or SLURM job per dataset
and model ([Running on a cluster](../guide/running.md#57-running-on-a-cluster)). `run/default_runs.sh` lists
every experiment in the paper.
