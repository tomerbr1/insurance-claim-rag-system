"""
Utilities to silence noisy NLTK downloader logs.

LlamaIndex may trigger downloads of NLTK resources (e.g., punkt_tab) during
tokenization. The default NLTK downloader prints status messages to stdout,
which clutters startup logs. This helper wraps the downloader with quiet=True
so required resources can still be fetched without noisy output.
"""

import functools
from typing import Callable


def _quiet_download(download_fn: Callable, *args, **kwargs):
    """Wrap NLTK's download function to enforce quiet mode."""
    kwargs["quiet"] = True
    return download_fn(*args, **kwargs)


def silence_nltk_downloads():
    """Force NLTK downloads into quiet mode; safe to call multiple times."""
    try:
        import nltk

        nltk.download = functools.partial(_quiet_download, nltk.download)
    except Exception:
        # If NLTK is not installed or can't be modified, fail silently.
        return

