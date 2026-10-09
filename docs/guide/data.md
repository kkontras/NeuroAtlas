# 2. Data

The benchmarks run on 42 public and restricted EEG datasets, and none of them ship with the
package. Open datasets are downloaded by the tool. The others need an account, a signed agreement
or a request to their owners, and the tool tells you where to ask and where to put the files. This
chapter shows how to see which datasets you have, how to get the others, and where they go.

## 2.1 Access

| Access | Datasets |
|---|---|
| Open, downloaded with `neuroatlas data download` | Sleep-EDF Expanded, HMC, UCDDB, DOD, Siena, CHB-MIT, Helsinki Neonatal, Bonn, EEGMat, ArithmeticTask, 14 MOABB BCI datasets |
| [NSRR](https://sleepdata.org) account, then `neuroatlas config token nsrr` | CFS, HomePAP, MESA, MrOS, SHHS, STAGES, WSC |
| Request from the data owners | DCSM, ISRUC, MASS, PhysioNet 2026, NMT, EPILEPSIAE, TUSZ, TUAB, DREAMER |
| Available from the authors | SeizeIT1, SeizeIT2 |

Every dataset has its own folder, `<data root>/<dataset>`. This is where `data download` puts it
and where `run` looks for it. MOABB datasets are the exception and use the MOABB folder
([MOABB datasets](#24-moabb-datasets)). If you already have a dataset, point to it with
`neuroatlas config set <dataset>.data_root DIR` ([Datasets you already have](configuration.md#12-datasets-you-already-have)).

`neuroatlas list datasets` lists every dataset, and `--grep sleep` narrows the list. The usual
start is a benchmark's datasets:

```bash
neuroatlas data status sleep_stage
neuroatlas data download sleep_edf_expanded --dry-run
neuroatlas data download sleep_edf_expanded --mirror aws
```

## 2.2 Checking what you have

`data status` takes benchmarks, domains or dataset names. It looks for each dataset exactly where
a run will.

```text
$ neuroatlas data status epilepsy
dataset            host                    state                  download
bonn               web                     found (500/500 files)  downloadable
chbmit             Zenodo                  missing                downloadable
epilepsiae         campus-technologies.de  missing                manual
helsinki_neonatal  Zenodo                  found (79/79 files)    downloadable
nmt                dll.seecs.nust.edu.pk   missing                manual
siena              Zenodo                  found (41/41 files)    downloadable
sz1                the authors             missing                from the authors
sz2                the authors             missing                from the authors
tuab               TUH                     missing                credentialed
tusz               TUH                     missing                credentialed

10 datasets: 7 missing, 3 found
...
not here; where each was looked for, and the setting that points it at a copy elsewhere:
  chbmit      $DATA/chbmit      chbmit.bids_root
  epilepsiae  $DATA/epilepsiae  epilepsiae.data_root
  ...
  tusz        $DATA/tusz        tusz.raw_root
$DATA = <data root>
```

Under the table, the tool explains the words it used and prints the `fix:` lines that apply. Add
`-v` to see every path and setting, and whether a channel map ships for the dataset.

The `state` column says whether the data is here:

| State | Meaning |
|---|---|
| `found (n/N files)` | Every expected file is there. Shows `found (n files)` when the total is not known. |
| `partial (n/N files)` | Fewer files than a complete copy, for example after a download that stopped part-way. `run` refuses it unless you pass `--allow-partial`. |
| `empty` | The folder exists but holds no recordings. `run` refuses it too. |
| `missing` | The folder does not exist. |
| `not downloaded` | A MOABB dataset that is not in the MOABB folder. |
| `not prepared` | The preprocessed file this dataset is read from is not there. |
| `prepared` | The preprocessed file is there. |
| `not configured` | No data root is set. Run `neuroatlas config set data_root DIR`. |
| `no path` | No folder is recorded for this dataset, so it cannot be checked. |
| `unknown` | A package its reader needs is not installed. The line under the row gives the install command. |

The `download` column says how to get it:

| Download | Meaning |
|---|---|
| `downloadable` | `data download` fetches it. No account needed. |
| `credentialed` | You need an approved account or a signed agreement first. For NSRR, `data download` then fetches it with your token. TUH data you download yourself. |
| `manual` | There is no download API. `data download` tells you where to request it and where to put it. |
| `from the authors` | Not public. Ask the authors. `data download` tells you where to put it. |

## 2.3 Downloading

Run `data download` with one or more dataset names. Add `--dry-run` first to see the plan. It
makes every check a real run makes and transfers nothing. What happens next depends on where the
dataset is served from:

| Host | Datasets | What happens |
|---|---|---|
| PhysioNet | Sleep-EDF Expanded (8.1 GB), UCDDB (1.3 GB), HMC (15.7 GB), EEGMat (0.18 GB) | `wget` from physionet.org, resumable. Add `--mirror aws` to use PhysioNet's open copy on AWS, which is much faster. |
| Zenodo | CHB-MIT (21.7 GB), Siena (4.5 GB), Helsinki Neonatal (4.3 GB), DOD (58.1 GB) | Resumable and md5-checked. Zip files are unpacked and then deleted. Keep them with `--keep-archive`. You need about twice the size free while a zip exists. |
| web | Bonn (3 MB), ArithmeticTask (0.7 GB) | Bonn: five zip files from the University of Bonn's site. ArithmeticTask: one zip from OSF holding one zip per experiment. Both md5-checked and unpacked. |
| NSRR | CFS, HomePAP, MESA, MrOS, SHHS (375 GB), STAGES, WSC | The official `nsrr` tool, after a check for 50 GB of free disk (for SHHS, its 375 GB). Needs your NSRR token. Each study's `datasets/` folder comes with the recordings: its participant tables, which give brain age the participants' ages. |
| MOABB | the 14 MOABB BCI datasets | MOABB's own download into the MOABB folder. Needs the [BCI install lines](../getting-started/installation.md#extra-packages). |
| TUH, and sites without an API | TUAB, TUSZ, DCSM, ISRUC, MASS, NMT, EPILEPSIAE, PhysioNet 2026, DREAMER | Prints where to request the data and where to put it. |
| the authors | SeizeIT1, SeizeIT2 | Prints where to put the files once you have them. |

Each download prints a header, one line per file, and a result line:

```text
$ neuroatlas data download bonn
downloading 1 dataset: bonn
[1/1] bonn: downloading from www.ukbonn.de into <data root>/bonn
  [1/5] z.zip: 591 kB in 0s, md5 ok
  ...
[1/1] bonn: downloaded, 5 files, 3.2 MB (1s)
```

Run the same command again and it skips the files that are already complete. Stop a download
with Ctrl-C and the next run continues where it stopped. A download refused before it started (no
token, not enough disk, a missing tool) exits with status 2. A transfer that failed exits with 1.

Measured download times: Sleep-EDF Expanded from the AWS copy took 18 min (399 files, 8.2 GB).
UCDDB from physionet.org came at about 100 KB/s and took 3 h 35 min for 1.3 GB, so use
`--mirror aws`. Helsinki Neonatal from Zenodo took 13 min. BNCI2014_004 from MOABB took 1 min.

### A few recordings first

To try a dataset before the full download, fetch only its first few recordings:

```bash
neuroatlas data download ucddb --first 2
```

This is enough for `check` to read real data. `run` needs the whole dataset. `--first` works for
PhysioNet, NSRR and file-by-file Zenodo downloads.

### NSRR datasets

The NSRR datasets are fetched with the `nsrr` tool, a Ruby gem. Install Ruby with its development
headers (`ruby-dev` or `ruby-devel`), then install the gem and put its folder on your `PATH`:

```bash
gem install --user-install nsrr irb
export PATH="$(ruby -e 'print Gem.user_dir')/bin:$PATH"
```

`data download shhs` fetches the EDFs of both SHHS visits (8444 recordings), NSRR's annotations
and the dataset tables. The reader takes the C4-A1 EEG, resampled to 100 Hz and band-passed
0.3-40 Hz, and scores each 30 s epoch from the annotations. Stage 4 counts as N3, unscored and
movement epochs are left out, and wake beyond the largest sleep stage is trimmed from the ends of
the night. Ages come from the `shhs1-dataset` and `shhs2-dataset` tables, each participant's age
at that visit.

## 2.4 MOABB datasets

MOABB keeps all its datasets in one folder. The tool picks it in this order:

1. `$MNE_DATA`
2. `MNE_DATA` in MNE's config file, `~/.mne/mne-python.json` (read, never written)
3. `<data root>/mne_data`
4. `~/mne_data`

`data download`, `data status` and `run` all use the same folder. `config show` prints which one
and why. To move it, set `export MNE_DATA=/your/mne_data`.

There is nothing to prepare first. `run` reads the MOABB recordings directly and cuts the trials
itself, in each model's band, sampling rate and notch, with each dataset's trial window. By
default that is confound filtering: for motor imagery 4-40 Hz at the model's rate, average
reference, and the trial from 1 s to 4 s after the cue. [BCI variants](benchmarks.md#bci-variants)
lists the others.

`list datasets --all` also shows MOABB datasets outside the paper. Most of them need a newer
moabb than 1.2.0, which needs numpy 2. The tool refuses those and says so.

## 2.5 Faster copies of epilepsy datasets

No dataset needs a build step. Five epilepsy datasets (Bonn, EPILEPSIAE, SeizeIT1, TUAB and TUSZ)
can be converted into a copy that is faster to read:

```bash
neuroatlas data prepare --list
neuroatlas data prepare tuab
```

The copy goes to `<cache root>/prepared/`. `data prepare` refuses to start until the raw data is
there. `--dry-run` prints the build command, `--dest` writes elsewhere and `--shard K/N` builds
one shard.

## 2.6 Files built from the raw data

The four `bci_cognitive` datasets are read from files that `data prepare` builds from their raw
data:

```bash
neuroatlas data download eegmat          # PhysioNet, 180 MB
neuroatlas data prepare eegmat
```

ArithmeticTask downloads the same way (OSF, 0.7 GB). DREAMER is on Zenodo, where access is
granted on request: `data download dreamer_valence` says where to ask and where to put
`DREAMER.mat`. `dreamer_valence` and `dreamer_arousal` read the same file, so put a copy or a link
in each folder.

`data prepare` writes two files per model format into `<cache root>/prepared/<dataset>/`, each at
the format's rate with a 50 Hz notch and the average reference.
`<name>_preprocessed_trackD_<format>.pkl` is filtered 4-40 Hz, and the default (confound
filtering) reads it. `<name>_preprocessed_<format>.pkl` is in the format's band, and the
`no_filtering` variants read it. The formats are `labram` (0.1-75 Hz, 200 Hz), `bendr`
(0.5-70 Hz, 256 Hz, read by EEGPT) and `steegformer` (0.1-64 Hz, 128 Hz, read by ST-EEGFormer and
SleepFM). Each model reads its format's file. EEGMat and DREAMER are cut into 4 s windows,
ArithmeticTask into 1 s windows.

`data status` shows `not prepared` until the files are built, with the command that builds them.
A file whose recorded band is the other filtering's is refused, with the command that runs the
variant it belongs to. The `preprocessed_path` setting points at a file built elsewhere.
