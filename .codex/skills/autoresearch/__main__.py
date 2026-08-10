# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Module execution (`python -m skills.autoresearch`) delegates to the CLI.
from skills.autoresearch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
