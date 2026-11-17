"""Root folder for thesis data (not code).

Default is Geron's layout, D:/Thesis. On a laptop without a D: drive, set the
environment variable THESIS_ROOT to another folder (e.g. C:/Users/<you>/Thesis).
Everything below it keeps the same layout: baguio_corpus_sample/, baguio_annotation/,
ground_truth/final/, pipeline_cache/doctr_dev/, pipeline_checks/.
"""
import os

THESIS = os.environ.get('THESIS_ROOT', r'D:\Thesis')


def thesis_path(*parts):
    """Path under the thesis data root."""
    return os.path.join(THESIS, *parts)
