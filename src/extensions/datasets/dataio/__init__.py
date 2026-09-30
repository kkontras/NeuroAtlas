"""EEGBenchmarks dataset namespace.

Keep this package import-light so benchmark registry and adapter imports do not
pull heavy optional dataset dependencies unless a specific dataset module is
requested explicitly.

Demographic keys
----------------
A dataio publishes whichever of these it has access to on each batch's meta
dict; consumers treat a missing key as absent rather than as a default.
Surveyed across the 8 demographics-aware dataios (epilepsiae, nmt, tuab,
tusz, helsinki_neonatal, siena_bids, chbmit_bids, aub_med):

    age                     int, -1 = missing
    gender                  str, "m"/"f"/"" (lowercase)
    age_years               float (AUB-Med only, preserves 4.5y)
    onset_age               int (Epilepsiae)
    hospital                str (Epilepsiae -- UKLFR/HUC/HdlPS)
    focus_localisation      str (Epilepsiae)
    bw_grams                str (Helsinki Neonatal -- "less than 2500g", etc.)
    gestational_age_weeks   str (Helsinki Neonatal)
    eeg_to_pma_weeks        str (Helsinki Neonatal)
    diagnosis               str (Helsinki Neonatal)
    neuroimaging            str (Helsinki Neonatal)
    primary_localisation    str (Helsinki Neonatal)
    comment                 str (CHB-MIT BIDS -- participants.tsv free-text)

This list is documentation, not a contract enforced in code: the helpers that
once collected these into npz columns and per-subject CSVs had no callers and
were removed, leaving this as the only written record.
"""

__all__ = [
    "parkinson",
    "shhs",
    "sleep_edf",
]
