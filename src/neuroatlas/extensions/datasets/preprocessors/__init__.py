"""Builders that turn a raw corpus into the cache an adapter reads.

Reached two ways, which is why they live here and not under entrypoints:
`prepare` dispatches to one by the `pipeline.preprocessor.module` path in a
cohort file, and several adapters import them directly as libraries. They
are not verbs -- `prepare` is the verb.

Ten cohorts declare a builder, but only the five MOABB motor-imagery sets
require it: MOABB has to download and epoch the corpus before anything can
read it. The other five (bonn, epilepsiae, sz1, tuab, tusz) mark theirs
`optional: true` -- their adapters read the raw EDF directly, and the cache
only speeds up large sweeps, so `prepare` is never a prerequisite for
`embed` there.
"""
