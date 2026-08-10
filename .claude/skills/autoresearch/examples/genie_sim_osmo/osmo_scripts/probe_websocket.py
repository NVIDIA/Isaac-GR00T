#!/usr/bin/env python3
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
