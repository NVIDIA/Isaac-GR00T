# Exported OSS tests

This directory contains tests that ship with the OSS package. They must run
using only files present in the exported `oss/` tree and must not depend on the
layout of its source monorepo.

Repository assembly checks, including assertions about whether paths under
`oss/` are symlinks, belong in the internal `tests/oss/` directory instead.
