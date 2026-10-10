# SPDX-License-Identifier: AGPL-3.0-only
"""Reuse viz unchanged with web-owned Windows scratch directories.

Python 3.12's Windows TemporaryDirectory uses a restrictive 0700 ACL that
cannot be reopened by this managed process. Use inherited ACLs inside the
already private job directory; never change tempfile or viz process globals.
"""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace
import uuid

from roadscope import viz


@contextmanager
def job_scratch(*, prefix, dir):
    parent = Path(dir).resolve(strict=True)
    path = parent / (prefix + uuid.uuid4().hex)
    path.mkdir()  # Inherit the enclosing job directory's ACL.
    try:
        yield str(path)
    finally:
        # Delete only our generated directory, never an arbitrary caller path.
        if path.resolve().parent != parent or path.is_symlink():
            raise RuntimeError('Scratch directory escaped its job directory')
        shutil.rmtree(path)


def with_job_scratch(function):
    if os.name != 'nt':
        return function
    namespace = {**function.__globals__, 'tempfile': SimpleNamespace(TemporaryDirectory=job_scratch)}
    local = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    local.__kwdefaults__ = function.__kwdefaults__
    return wraps(function)(local)


prepare_upload = with_job_scratch(viz.prepare_upload)
render_video = with_job_scratch(viz.render_video)
render_timeline = viz.render_timeline
