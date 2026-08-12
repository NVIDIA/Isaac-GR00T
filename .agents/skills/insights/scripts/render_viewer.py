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

"""Render a static, human-facing viewer for the published ``oss/.insights/`` projection.

Code and data are kept separate: this script (and the ``graph.html`` shell +
``vis-network.min.js`` it emits) lives in ``oss/skills/insights/scripts/``, while the
published projection it renders lives in ``oss/.insights/``. It reads **only** that
published projection (``index.json`` + ``nodes/*.json``) — never the internal knowledge
graph — so it structurally cannot expose anything the publish gate withheld (concepts,
edges, provenance, or any ``restricted`` content).

The viewer is split into two files so the node/edge data is **loaded dynamically when
the page opens** instead of being baked into the HTML:

  * ``graph.html``      — a small, static shell (header, legend, filters, layout), in
                          ``oss/skills/insights/scripts/``.
  * ``graph_data.json`` — the ``{nodes, edges}`` payload the page ``fetch()``es at load,
                          in ``oss/.insights/``.

Because ``graph.html`` fetches its data at runtime, browsers block that request under
``file://``. Serve the ``oss/`` dir (so the cross-folder data fetch resolves) and view it::

    cd oss
    python3 -m http.server 8000
    # then open http://localhost:8000/skills/insights/scripts/graph.html

The JS library is vendored locally (``vis-network.min.js``, no CDN). The rendering style
mirrors ``scripts/visualize_knowledge_graph.py`` (legend, hover tooltips, node-type
filters, fit-view button, fixed radial layout). Because the projection deliberately drops
graph edges, the only relationships drawn are those recoverable from the public data
itself:
  * topic  -> finding / evidence   ("about", bucket membership)
  * finding -> evidence            ("cites", from the shared id suffix
                                    ``finding:doc:X`` <-> ``doc:X``)

Regenerate after every export::

    python3 render_viewer.py

The output is a snapshot; if the projection changes and this is not re-run, the viewer
will lag.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Code lives in oss/skills/insights/scripts/; the data it renders lives in the OSS
# projection dir oss/.insights/. Paths resolve relative to __file__ so the viewer works
# regardless of CWD.
SCRIPTS_DIR = Path(__file__).resolve().parent  # oss/skills/insights/scripts
DATA_DIR = SCRIPTS_DIR.parents[2] / ".insights"  # oss/.insights
VIS_JS = "vis-network.min.js"
OUTPUT = SCRIPTS_DIR / "graph.html"  # the viewer shell (code)
DATA_OUTPUT = DATA_DIR / "graph_data.json"  # the fetched payload (data)
# Relative URL the served graph.html uses to fetch its data. graph.html is at
# oss/skills/insights/scripts/ and the payload at oss/.insights/, so serve the oss/ dir
# (see the module docstring) and this resolves to oss/.insights/graph_data.json.
DATA_URL = "../../../.insights/graph_data.json"

# A harmonious, well-separated palette: indigo hub, teal-green claims, orange
# support — distinct hues with a layered feel. Edges use light tints of the endpoint
# they relate to (about -> topic's indigo, cites -> evidence's orange) so they read as
# structure without overpowering the nodes.
NODE_COLORS = {
    "topic": "#6741d9",  # indigo — the central hub
    "finding": "#099268",  # teal-green — distilled claims
    "evidence": "#fd7e14",  # orange — raw citable passages (support)
}

EDGE_COLORS = {
    "about": "#b197fc",  # light indigo — topic membership
    "cites": "#ffa94d",  # light orange — finding -> its source chunk
}

NODE_SIZES = {"topic": 38, "finding": 17, "evidence": 11}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_projection() -> dict:
    index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    buckets = {}
    for topic in index["topics"]:
        buckets[topic] = json.loads(
            (DATA_DIR / "nodes" / f"{topic}.json").read_text(encoding="utf-8")
        )
    return {"index": index, "buckets": buckets}


# Radial layout tunables (positions are fixed; physics is off in the viewer).
FIND_R0 = 240  # radius of the innermost finding ring
RING_GAP = 150  # radial distance between successive rings
BASE_CAP = 14  # nodes on the innermost ring
CAP_INC = 8  # extra node capacity per outer ring
CHUNK_GAP = 160  # gap between the outermost finding ring and the first chunk ring
CLUSTER_R = 3400  # radius of the circle the topic centers sit on


def _ring_positions(count: int, r0: float) -> list[tuple[float, float]]:
    """Distribute ``count`` items over concentric rings starting at radius ``r0``.

    Returns ``(radius, angle)`` per item and the total radial extent used, so the
    caller can stack the next zone (chunks) outside the last ring.
    """
    positions: list[tuple[float, float]] = []
    placed, ring = 0, 0
    while placed < count:
        cap = BASE_CAP + ring * CAP_INC
        radius = r0 + ring * RING_GAP
        k = min(cap, count - placed)
        for m in range(k):
            angle = 2 * math.pi * m / k + ring * 0.35  # stagger rings so spokes offset
            positions.append((radius, angle))
        placed += k
        ring += 1
    return positions


def _ring_extent(count: int, r0: float) -> float:
    rings = 0
    placed = 0
    while placed < count:
        placed += BASE_CAP + rings * CAP_INC
        rings += 1
    return r0 + max(0, rings - 1) * RING_GAP


def build_vis_data(projection: dict) -> tuple[list[dict], list[dict]]:
    """Lay the projection out as a forest of radial trees.

    Each topic that has
    findings: the topic at the centre, its findings on inner rings, and the chunks
    those findings cite on outer rings ("support"). Positions are fixed polar
    coordinates so the onion structure is stable without physics.
    """
    buckets = projection["buckets"]
    all_chunks = {c["chunk_id"]: c for b in buckets.values() for c in b["chunks"]}
    topics = [t for t in sorted(buckets) if buckets[t]["findings"]]

    vis_nodes: list[dict] = []
    vis_edges: list[dict] = []
    seen: set[str] = set()

    n = len(topics)
    for ti, topic in enumerate(topics):
        if n == 1:
            cx, cy = 0.0, 0.0
        else:
            a = 2 * math.pi * ti / n - math.pi / 2
            cx, cy = CLUSTER_R * math.cos(a), CLUSTER_R * math.sin(a)

        findings = buckets[topic]["findings"]
        topic_id = f"topic:{topic}"
        vis_nodes.append(
            {
                "id": topic_id,
                "label": topic,
                "group": "topic",
                "title": f"topic · {topic}\n{len(findings)} findings",
                "x": round(cx),
                "y": round(cy),
                "size": NODE_SIZES["topic"],
                "font": {"size": 22, "color": "#6741d9"},
            }
        )
        seen.add(topic_id)

        find_pos = _ring_positions(len(findings), FIND_R0)
        chunk_r0 = _ring_extent(len(findings), FIND_R0) + CHUNK_GAP
        chunk_pos_iter = iter(_ring_positions(len(findings), chunk_r0))

        for j, f in enumerate(findings):
            fid = f["id"]
            r, ang = find_pos[j]
            fx, fy = cx + r * math.cos(ang), cy + r * math.sin(ang)
            tooltip = f"finding · {fid}\n{f.get('title', '')}"
            if f.get("body"):
                tooltip += f"\n\n{_truncate(f['body'], 240)}"
            vis_nodes.append(
                {
                    "id": fid,
                    "label": _truncate(f.get("title") or fid, 28),
                    "group": "finding",
                    "title": tooltip,
                    "x": round(fx),
                    "y": round(fy),
                    "size": NODE_SIZES["finding"],
                }
            )
            seen.add(fid)
            vis_edges.append(
                {
                    "from": topic_id,
                    "to": fid,
                    "title": "about",
                    "color": {"color": EDGE_COLORS["about"], "opacity": 0.55},
                    "arrows": "to",
                }
            )

            chunk_id = fid[len("finding:") :] if fid.startswith("finding:") else None
            c = all_chunks.get(chunk_id) if chunk_id else None
            if c and chunk_id not in seen:
                cr, cang = next(chunk_pos_iter)
                chx, chy = cx + cr * math.cos(cang), cy + cr * math.sin(cang)
                ct = f"evidence · {chunk_id}\n{c.get('title', '')}"
                if c.get("text"):
                    ct += f"\n\n{_truncate(c['text'], 240)}"
                vis_nodes.append(
                    {
                        "id": chunk_id,
                        "label": _truncate(c.get("title") or chunk_id, 28),
                        "group": "evidence",
                        "title": ct,
                        "x": round(chx),
                        "y": round(chy),
                        "size": NODE_SIZES["evidence"],
                    }
                )
                seen.add(chunk_id)
                vis_edges.append(
                    {
                        "from": fid,
                        "to": chunk_id,
                        "title": "cites",
                        "color": {"color": EDGE_COLORS["cites"], "opacity": 0.65},
                        "arrows": "to",
                    }
                )
    return vis_nodes, vis_edges


def render_html(vis_nodes: list[dict], vis_edges: list[dict], *, source: str) -> str:
    """Render the static viewer shell.

    The node/edge arrays are **not** inlined here;
    the page fetches them from ``graph_data.json`` at load time. Only small, derived
    summaries (legend counts, filters, group colors) are baked in.
    """
    node_legend = "".join(
        f'<span class="chip"><i style="background:{c}"></i>{t} '
        f"({sum(1 for n in vis_nodes if n['group'] == t)})</span>"
        for t, c in NODE_COLORS.items()
        if any(n["group"] == t for n in vis_nodes)
    )
    edge_present = {e["title"] for e in vis_edges}
    edge_legend = "".join(
        f'<span class="chip"><i style="background:{EDGE_COLORS.get(t, "#ced4da")}"></i>'
        f"{t} ({sum(1 for e in vis_edges if e['title'] == t)})</span>"
        for t in EDGE_COLORS
        if t in edge_present
    )
    filters = "".join(
        f'<label><input type="checkbox" class="nf" value="{t}" checked> {t}</label>'
        for t in NODE_COLORS
        if any(n["group"] == t for n in vis_nodes)
    )
    # Pin each group's color explicitly so the on-graph node colors match the legend;
    # otherwise vis-network assigns groups colors from its own palette and overrides
    # the per-node color, desyncing the legend from the graph.
    groups_json = json.dumps({t: {"color": c} for t, c in NODE_COLORS.items()})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GR00T knowledge graph — .insights/</title>
<script src="{VIS_JS}"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }}
  header {{ padding: 10px 14px; border-bottom: 1px solid #dee2e6; }}
  h1 {{ font-size: 16px; margin: 0 0 6px; }}
  .meta {{ color: #868e96; font-size: 12px; }}
  .legend {{ margin-top: 8px; font-size: 12px; }}
  .chip {{ display: inline-block; margin: 2px 10px 2px 0; }}
  .chip i {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px;
            margin-right: 4px; vertical-align: middle; }}
  .controls {{ margin-top: 6px; font-size: 12px; }}
  .controls label {{ margin-right: 10px; }}
  #net {{ width: 100%; height: calc(100vh - 168px); }}
</style>
</head>
<body>
<header>
  <h1>GR00T knowledge graph — .insights/</h1>
  <div class="meta">{len(vis_nodes)} nodes · {len(vis_edges)} edges · source: {source}
    · data loaded at runtime from <code>graph_data.json</code>
    · radial tree: topic at centre → finding rings → support (chunk) rings
    · drag to pan, scroll to zoom, hover a node for its title</div>
  <div class="legend"><b>nodes</b> &nbsp; {node_legend}</div>
  <div class="legend"><b>edges</b> &nbsp; {edge_legend}</div>
  <div class="controls"><b>show</b>
    {filters}
    &nbsp;|&nbsp; <button id="fit">fit view</button>
  </div>
</header>
<div id="net"></div>
<script>
  // #1 / #2: allNodes and allEdges are loaded dynamically when the page opens,
  // fetched from graph_data.json rather than baked into this HTML.
  const DATA_URL = "{DATA_URL}";
  const groups = {groups_json};
  let allNodes = [];
  let allEdges = [];

  function render(data) {{
    allNodes = data.nodes || [];
    allEdges = data.edges || [];
    const nodes = new vis.DataSet(allNodes);
    const edges = new vis.DataSet(allEdges);
    const network = new vis.Network(
      document.getElementById("net"),
      {{ nodes, edges }},
      {{
        groups: groups,
        nodes: {{ shape: "dot", size: 12, font: {{ size: 12 }} }},
        edges: {{ smooth: {{ type: "continuous" }}, width: 1 }},
        physics: false,
        interaction: {{ hover: true, tooltipDelay: 120 }}
      }}
    );
    network.fit();
    document.getElementById("fit").onclick = () => network.fit();

    const hidden = new Set();
    document.querySelectorAll(".nf").forEach((box) => {{
      box.onchange = () => {{
        if (box.checked) hidden.delete(box.value); else hidden.add(box.value);
        nodes.update(
          allNodes.map((n) => ({{ id: n.id, hidden: hidden.has(n.group) }}))
        );
      }};
    }});
  }}

  function showError(msg) {{
    document.getElementById("net").innerHTML =
      '<div style="padding:20px;max-width:760px;color:#495057;font-size:14px;line-height:1.5">' +
      '<p style="color:#c92a2a;font-weight:600">Could not load graph data (' + msg + ').</p>' +
      '<p>This viewer <b>fetches its data at runtime</b> from <code>graph_data.json</code>, ' +
      'so most browsers block it when the file is opened directly via <code>file://</code>. ' +
      'Serve the <code>oss/</code> dir instead (the data lives in <code>oss/.insights/</code>):</p>' +
      '<pre style="background:#f1f3f5;padding:10px;border-radius:6px">' +
      'cd oss\\npython3 -m http.server 8000</pre>' +
      '<p>then open <code>http://localhost:8000/skills/insights/scripts/graph.html</code>.</p></div>';
  }}

  fetch(DATA_URL, {{ cache: "no-store" }})
    .then((r) => {{ if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }})
    .then(render)
    .catch((e) => showError(e.message || e));
</script>
</body>
</html>
"""


def main() -> None:
    projection = load_projection()
    vis_nodes, vis_edges = build_vis_data(projection)

    payload = {
        "schema_version": "groot_insights_viewer_data.v0",
        "source": ".insights/",
        "generated_from": "index.json + nodes/*.json",
        "nodes": vis_nodes,
        "edges": vis_edges,
    }
    DATA_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    html = render_html(vis_nodes, vis_edges, source=".insights/")
    OUTPUT.write_text(html, encoding="utf-8")

    print(f"wrote {DATA_OUTPUT} ({len(vis_nodes)} nodes, {len(vis_edges)} edges)")  # noqa: T201
    print(f"wrote {OUTPUT}")  # noqa: T201
    if not (SCRIPTS_DIR / VIS_JS).exists():
        print(  # noqa: T201
            f"WARNING: {VIS_JS} not found next to the viewer — the graph will not render."
        )


if __name__ == "__main__":
    main()
