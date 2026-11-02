#!/usr/bin/env python
"""Fault-injection check for the Stage 3 fail-loud path.

AIDRS must abort when one library cannot be quantified: a silently dropped
sample disappears from the count/CPM matrices *and* from the CPM
denominator, so the run would exit 0 with a quietly smaller cohort.

This script reproduces that situation on a real run by deleting one
sample's ``{sample}_flnc_correct.ssc`` just before that sample is
quantified -- exactly the state the pipeline is in when bam2ssc failed to
produce (or the file was lost for) one library. The exception therefore
originates inside the *production* code path
(``IsoformQuantifier.quantify_sample`` -> ``FileNotFoundError``), not in
this harness.

Usage::

    E6_BROKEN_SAMPLE=<sample> python tools/repro/quant_fail_loud_check.py \\
        --reference <ref.fa> --bam <a.bam> <b.bam> --output <dir>

Run it twice: once with the default (expect a non-zero exit and
``aidrs.quantification_failures.tsv`` on disk) and once with
``--allow_partial_quantification`` (expect exit 0, a ``[WARN]`` line and
the same failure record).

NOTE: the ``if __name__ == "__main__":`` guard below is REQUIRED.
``src/gene_grouping.py`` and ``src/protein_coding_ability.py`` use
``mp.get_context("spawn")``, and a spawn child re-imports the parent's
``__main__`` module. Without the guard every worker would re-run the whole
pipeline (and fight over the output-directory lock).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.isoform_quantification import IsoformQuantifier  # noqa: E402

BROKEN = os.environ.get("E6_BROKEN_SAMPLE")
_orig_quantify_sample = IsoformQuantifier.quantify_sample


def _patched_quantify_sample(self, sample_name, transcript_model_df, output_dir):
    if sample_name == BROKEN:
        path = os.path.join(output_dir, "temp", f"{sample_name}_flnc_correct.ssc")
        if os.path.exists(path):
            os.remove(path)
            print(f"[FAULT-INJECT] removed {path}", flush=True)
    return _orig_quantify_sample(self, sample_name, transcript_model_df, output_dir)


IsoformQuantifier.quantify_sample = _patched_quantify_sample

from src.aidrs import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
