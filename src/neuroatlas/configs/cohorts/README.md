# Datasets

One row per dataset manifest. Each `cohort.yaml` holds every fact about its
dataset: where it comes from, its labels, how its folds are formed and its
reader's defaults. When this table and a manifest disagree, the manifest is
right.

50 datasets have a manifest: the 43 of the paper's evaluation
(`neuroatlas list datasets`) and 7 more the readers support.
`neuroatlas list datasets --all` adds 138 MOABB datasets outside the paper
(188 in all).

In the table, `source` is where the data comes from; `+N unused` counts the
label modes the reader accepts that no benchmark uses; *frozen* folds come
from a file shipped with the package, and *derived* folds from the reader's
seeded splitter.

## bci (22)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`arithmetic_task`](arithmetic_task/cohort.yaml) | yes | from the authors | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bi2013a`](bi2013a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bi2014a`](bi2014a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_001`](bnci2014_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_004`](bnci2014_004/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_008`](bnci2014_008/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2015_001`](bnci2015_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`cho2017`](cho2017/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreamer_arousal`](dreamer_arousal/cohort.yaml) | yes | manual | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreamer_valence`](dreamer_valence/cohort.yaml) | yes | manual | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreyer2023`](dreyer2023/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`eegmat`](eegmat/cohort.yaml) | yes | physionet | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`epflp300`](epflp300/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`erpcore2021_n170`](erpcore2021_n170/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`hinss2021`](hinss2021/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`kim2025betarange`](kim2025betarange/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`lee2019_mi`](lee2019_mi/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`liu2024`](liu2024/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`nakanishi2015`](nakanishi2015/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`physionet_mi`](physionet_mi/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`shin2017a`](shin2017a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`weibo2014`](weibo2014/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |

## epilepsy (11)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`aub_med`](aub_med/cohort.yaml) | — | mendeley | bipolar | 10.0 | `binary` | window | 6 <br><sub>derived</sub> | patient |
| [`bonn`](bonn/cohort.yaml) | yes | url | unipolar | — | `binary_s_vs_rest` <br><sub>+8 unused</sub> | recording | 5 <br><sub>derived</sub> | clip |
| [`chbmit`](chbmit/cohort.yaml) | yes | zenodo | bipolar | 10.0 | `binary` <br><sub>+1 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`epilepsiae`](epilepsiae/cohort.yaml) | yes | manual | bipolar | 10.0 | `binary` <br><sub>+2 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`helsinki_neonatal`](helsinki_neonatal/cohort.yaml) | yes | zenodo | unipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`nmt`](nmt/cohort.yaml) | yes | manual | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`siena`](siena/cohort.yaml) | yes | zenodo | unipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`sz1`](sz1/cohort.yaml) | yes | from the authors | unipolar | 10.0 | `binary` | window | 5 <br><sub>frozen</sub> | patient |
| [`sz2`](sz2/cohort.yaml) | yes | from the authors | unipolar | 10.0 | `binary` | window | 5 <br><sub>frozen</sub> | patient |
| [`tuab`](tuab/cohort.yaml) | yes | tuh | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`tusz`](tusz/cohort.yaml) | yes | tuh | bipolar | 10.0 | `binary` <br><sub>+6 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |

## sleep (17)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`cfs`](cfs/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`dcsm`](dcsm/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`dod`](dod/cohort.yaml) | yes | zenodo | — | — | `sleep_stage`, `osa_group` | sleep_stage: window, osa_group: recording | 5 <br><sub>frozen</sub> | patient |
| [`hmc`](hmc/cohort.yaml) | yes | physionet | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`hpap_lab_full`](hpap_lab_full/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`isruc`](isruc/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `diagnosis`, `has_diagnosis`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, diagnosis: recording, has_diagnosis: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`mass`](mass/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`mesa`](mesa/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`mros`](mros/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`parkinson`](parkinson/cohort.yaml) | — | from the authors | — | — | `group` | recording | 5 <br><sub>derived</sub> | patient |
| [`physionet2026`](physionet2026/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `ci_label`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, ci_label: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`shhs`](shhs/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`sleep_edf`](sleep_edf/cohort.yaml) | — | physionet | — | — | `sleep_stage` | window | — | patient |
| [`sleep_edf_expanded`](sleep_edf_expanded/cohort.yaml) | yes | physionet | — | — | `sleep_stage`, `condition`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, condition: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`stages`](stages/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`ucddb`](ucddb/cohort.yaml) | yes | physionet | — | — | `sleep_stage` <br><sub>+1 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`wsc`](wsc/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |
