#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Return success only when the policy server accepts a websocket upgrade."""

from __future__ import annotations

import base64
import http.client
import os
import sys


def main() -> None:
    host, port = sys.argv[1], int(sys.argv[2])
    conn = http.client.HTTPConnection(host, port, timeout=2)
    conn.request(
        "GET",
        "/",
        headers={
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode("ascii"),
            "Sec-WebSocket-Version": "13",
        },
    )
    raise SystemExit(conn.getresponse().status != 101)


if __name__ == "__main__":
    main()
