#!/usr/bin/env python3
"""Compact query-toggle review of furnished and floorplan depth evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

from .hssd_furnished_depth_policy import AUDIT_DIRECTORY


def select_review_rows(rows: list[dict], *, per_scene: int = 12) -> list[dict]:
    """Show every exclusion/replacement and a few retained controls per scene."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)
    selected: list[dict] = []
    order = {"fail": 0, "review": 1, "pass": 2}
    for scene in sorted(grouped):
        ranked = sorted(grouped[scene], key=lambda row: (
            order.get(str(row.get("decision")), 3),
            -int((row.get("counts") or {}).get("reliable_no_hit", 0)),
            -int(sum((row.get("counts") or {}).get(key, 0) for key in (
                "false_closed", "false_open", "target_disagreement", "unexplained_no_hit",
            ))),
            int(row["sample_index"]),
        ))
        if len(ranked) <= per_scene:
            picked = ranked
        else:
            picked = [row for row in ranked if row.get("decision") != "pass" or
                      row.get("source_kind") == "replacement"]
            controls = [row for row in ranked if row.get("decision") == "pass" and
                        row.get("source_kind") != "replacement"]
            picked.extend(controls[:3])
            ordinary = next((row for row in controls if not row.get("warning_reasons")), None)
            if ordinary is not None and ordinary not in picked:
                picked.append(ordinary)
        selected.extend(picked)
    return selected


def write_review(path: Path, rows: list[dict], dataset_root: Path) -> None:
    """Write a self-contained index; RGB and maps stay in their canonical paths."""

    dataset_root = Path(dataset_root)
    path = Path(path)
    selected = select_review_rows(rows)
    compact = []
    for row in selected:
        scene = str(row["scene_id"])
        image_path = str(row.get("image_path") or "")
        if image_path.startswith(str(dataset_root)):
            image_path = str(Path(image_path).relative_to(dataset_root))
        compact.append({
            "scene": scene, "sample": int(row["sample_index"]), "split": row.get("split"),
            "decision": row.get("decision"), "reasons": row.get("reasons", []),
            "warnings": row.get("warning_reasons", []),
            "noHits": (row.get("counts") or {}).get("reliable_no_hit", 0),
            "sourceKind": row.get("source_kind", "original"),
            "ambiguous": row.get("ambiguous40"),
            "pose": row.get("pose_xyz_m"), "yaw": row.get("logical_yaw_deg"),
            "gridRow": row.get("grid_row"), "gridCol": row.get("grid_col"),
            "rgb": "../../../" + image_path,
            "map": f"../../../2DFP/{scene}/map.png",
            "furnished": row.get("furnished_depth40_m"),
            "target": row.get("target_depth40_m"),
            "exact": row.get("exact_map_depth40_m"),
            "stage": row.get("stage_depth40_m"),
        })
    payload = json.dumps(compact, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape("HSSD furnished depth ↔ 2DFP QA")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui;margin:16px;background:#f4f7fb;color:#18283e}}header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px}}select{{padding:6px}}main{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.card{{background:white;border:1px solid #d4deea;border-radius:8px;padding:12px;min-width:0}}img{{display:block;max-width:100%;height:auto}}#rgb{{width:100%}}#mapWrap{{position:relative;width:100%;max-width:850px}}#map{{width:100%}}#poseMarker{{position:absolute;inset:0;width:100%;height:100%}}.wide{{grid-column:1/-1}}svg{{width:100%;height:auto}}#plot text{{font:12px system-ui;fill:#53657d}}.legend{{display:flex;gap:18px;flex-wrap:wrap}}@media(max-width:900px){{main{{grid-template-columns:1fr}}.wide{{grid-column:auto}}}}</style></head>
<body><header><strong>{title}</strong><label>Scene <select id="sceneSelect"></select></label><label>Query <select id="querySelect"></select></label><span id="meta"></span></header>
<main><section class="card"><h3>Furnished RGB observation</h3><img id="rgb"></section><section class="card"><h3>2DFP with GT pose/yaw</h3><div id="mapWrap"><img id="map"><canvas id="poseMarker"></canvas></div><p>+X right; +Z down. Orange arrow = GT forward.</p></section>
<section class="card wide"><h3>40-ray camera-forward depth</h3><svg id="plot" viewBox="0 0 800 340" role="img" aria-label="Furnished and 2DFP depth curves"></svg><div class="legend"><span style="color:#d97706">━━ furnished</span><span style="color:#1764d2">━━ 2DFP target</span><span style="color:#4b8b6a">━━ exact map</span><span style="color:#9370b5">━━ bare stage diagnostic</span></div><p>RGB left → right; missing depth breaks a curve. Stage is diagnostic, not the acceptance reference.</p></section></main>
<script>const rows={payload},sceneSelect=document.getElementById('sceneSelect'),querySelect=document.getElementById('querySelect');
const scenes=[...new Set(rows.map(r=>r.scene))];sceneSelect.replaceChildren(...scenes.map(s=>new Option(s,s)));
function queries(){{const subset=rows.filter(r=>r.scene===sceneSelect.value);querySelect.replaceChildren(...subset.map((r,i)=>new Option(`Q${{String(r.sample).padStart(5,'0')}} · ${{r.sourceKind}} · ${{r.decision}} · no-hit ${{r.noHits}}/40`,String(i))));show();}}
function drawPose(r){{const img=document.getElementById('map'),canvas=document.getElementById('poseMarker');if(!img.naturalWidth)return;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);if(r.gridCol==null||r.gridRow==null)return;const a=r.yaw*Math.PI/180,x=r.gridCol,y=r.gridRow;c.strokeStyle='#f97316';c.fillStyle='#f97316';c.lineWidth=Math.max(3,canvas.width/220);c.beginPath();c.arc(x,y,Math.max(5,canvas.width/130),0,2*Math.PI);c.stroke();c.beginPath();c.moveTo(x,y);c.lineTo(x+Math.cos(a)*Math.max(18,canvas.width/28),y+Math.sin(a)*Math.max(18,canvas.width/28));c.stroke();}}
function plot(r){{const svg=document.getElementById('plot');svg.replaceChildren();const NS='http://www.w3.org/2000/svg',series=[['furnished','#d97706'],['target','#1764d2'],['exact','#4b8b6a'],['stage','#9370b5']];const values=series.flatMap(([k])=>r[k]||[]).filter(v=>v!=null&&Number.isFinite(v));const max=Math.max(2,Math.ceil(Math.max(...values,1)));for(let y=0;y<=max;y+=Math.max(1,Math.ceil(max/6))){{const line=document.createElementNS(NS,'line');line.setAttribute('x1','48');line.setAttribute('x2','778');line.setAttribute('y1',String(300-260*y/max));line.setAttribute('y2',String(300-260*y/max));line.setAttribute('stroke','#dce4ed');svg.append(line);const label=document.createElementNS(NS,'text');label.setAttribute('x','5');label.setAttribute('y',String(304-260*y/max));label.textContent=y+' m';svg.append(label);}}for(const [key,color] of series){{const a=r[key];if(!a)continue;let d='';a.forEach((v,i)=>{{if(v==null||!Number.isFinite(v))return;const x=48+730*i/39,y=300-260*v/max;d+=(i===0||a[i-1]==null?'M':'L')+x.toFixed(1)+','+y.toFixed(1);}});const path=document.createElementNS(NS,'path');path.setAttribute('d',d);path.setAttribute('stroke',color);path.setAttribute('fill','none');path.setAttribute('stroke-width','2.4');svg.append(path);}}}}
function show(){{const subset=rows.filter(r=>r.scene===sceneSelect.value),r=subset[Number(querySelect.value)];if(!r)return;document.getElementById('rgb').src=r.rgb;const map=document.getElementById('map');map.onload=()=>drawPose(r);map.src=r.map;document.getElementById('meta').textContent=`${{r.split}} · ${{r.decision}} · no-hit ${{r.noHits}}/40 · pose ${{(r.pose||[]).join(', ')}} · yaw ${{r.yaw}}° · diagnostics: ${{r.warnings.join(', ')||'none'}}`;plot(r);}}
sceneSelect.onchange=queries;querySelect.onchange=show;queries();</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args()
    audit_root = args.audit_root or args.dataset_root / "UnLoc" / AUDIT_DIRECTORY
    rows = [json.loads(line) for ledger in sorted((audit_root / "scenes").glob("*.jsonl"))
            for line in ledger.read_text(encoding="utf-8").splitlines()]
    for manifest in sorted((audit_root / "replacements").glob("*/replacement_manifest.json")):
        payload = json.loads(manifest.read_text())
        rows.extend({**row, "source_kind": "replacement"} for row in payload.get("records", [])
                    if row.get("accepted"))
    output = audit_root / "review" / "index.html"
    write_review(output, rows, args.dataset_root)
    print(json.dumps({"html": str(output), "scenes": len(set(r["scene_id"] for r in rows)),
                      "source_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
