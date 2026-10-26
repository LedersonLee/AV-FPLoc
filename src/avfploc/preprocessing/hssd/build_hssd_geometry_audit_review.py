#!/usr/bin/env python3
"""Build a compact remote-review gallery from the completed HSSD audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTROLS = (("105515151_173104068", 360), ("105515151_173104068", 874),
            ("103997586_171030666", 218), ("104862417_172226382", 8))


def _select_examples(rows: list[dict]) -> list[dict]:
    by_key = {(r["scene_id"], r["sample_index"]): r for r in rows}
    chosen = []
    keys = set()
    for key in CONTROLS:
        if key in by_key:
            chosen.append(by_key[key]); keys.add(key)
    categories = (
        ("False-closed no-hit", lambda r: r.get("counts", {}).get("false_closed_no_hit", 0)),
        ("False-closed hit", lambda r: r.get("counts", {}).get("false_closed_hit", 0)),
        ("False-open hit", lambda r: r.get("counts", {}).get("false_open_hit", 0)),
        ("Borderline review", lambda r: int(r["decision"] == "review") *
         (r.get("counts", {}).get("hit_mismatch", 0) + r.get("counts", {}).get("no_hit", 0))),
        ("Consistent pass", lambda r: int(r["decision"] == "pass")),
    )
    for label, score in categories:
        represented = set()
        candidates = sorted(rows, key=lambda r: (-score(r), r["scene_id"], r["sample_index"]))
        for row in candidates:
            key = (row["scene_id"], row["sample_index"])
            if score(row) <= 0 or key in keys or row["scene_id"] in represented:
                continue
            record = dict(row)
            record["review_category"] = label
            chosen.append(record)
            keys.add(key)
            represented.add(row["scene_id"])
            if len(represented) >= 8:
                break
    return chosen


def build_html(rows: list[dict], bounds: dict[str, dict]) -> str:
    items = []
    for row in rows:
        item = {k: row.get(k) for k in (
            "scene_id", "sample_index", "split", "decision", "reasons", "counts", "image_path",
            "pose_xyz_m", "logical_yaw_deg", "stage_depth40_m", "target_depth40_m", "review_category",
        )}
        item["bounds"] = bounds.get(str(row["scene_id"]))
        items.append(item)
    data = json.dumps(items, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HSSD UnLoc geometry audit review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#17243a;background:#f4f7fc}
body{margin:0;padding:20px;max-width:1700px;margin:auto}h1{font-size:22px;margin:0 0 8px}
p{margin:4px 0 12px;color:#536781}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
select{font:inherit;max-width:100%;padding:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.panel{background:white;border:1px solid #d0dded;border-radius:8px;padding:12px;min-width:0}
h2{font-size:16px;margin:0 0 8px}.media{width:100%;display:block}.map{position:relative}
.map img{width:100%;display:block;image-rendering:auto}.map svg{position:absolute;inset:0;width:100%;height:100%}
.legend{font-size:13px;color:#40536c;margin-top:6px}.depth{grid-column:1/-1}
svg.plot{width:100%;height:auto;display:block}.status{font-weight:600}
@media(max-width:800px){.grid{grid-template-columns:1fr}.depth{grid-column:1}}
</style></head><body>
<h1>HSSD UnLoc geometry audit — review examples</h1>
<p>Stage-only 3D depth vs. stored 2DFP camera-forward depth. A missing 3D hit is shown as a gap, not a zero-depth wall. Map: +X right, +Z down.</p>
<div class="controls"><label for="query">Query</label><select id="query"></select><span id="status" class="status"></span></div>
<div class="grid"><section class="panel"><h2>RGB observation</h2><img class="media" id="rgb" alt="Furnished RGB observation"><div id="pose" class="legend"></div></section>
<section class="panel"><h2>2DFP with GT pose and yaw</h2><div class="map"><img id="map" alt="Canonical training floorplan"><svg id="pose-svg" aria-label="GT pose overlay"></svg></div><div class="legend">White: indoor free; black: wall/exterior. Orange arrow: GT camera forward.</div></section>
<section class="panel depth"><h2>40-ray depth comparison</h2><svg id="plot" class="plot" role="img" aria-label="Stage 3D and 2DFP forward-depth curves" viewBox="0 0 1000 320"></svg><div class="legend"><span style="color:#0064d2">━━</span> 3D stage depth &nbsp; <span style="color:#c25a0b">━━</span> 2DFP target depth &nbsp; · · = 3D no-hit; horizontal axis: RGB left → right.</div></section></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const query=document.getElementById('query');
for(let i=0;i<data.length;i++){let r=data[i],o=document.createElement('option');o.value=i;o.textContent=`${r.review_category||'Control'} · ${r.scene_id} / Q${String(r.sample_index).padStart(5,'0')} · ${r.decision}`;query.append(o)}
const ns='http://www.w3.org/2000/svg';
function el(name,attrs={}){let x=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,v);return x}
function plotCurve(svg,values,color){let path=[];for(let i=0;i<40;i++){let v=values?.[i];if(v===null||v===undefined||!Number.isFinite(v)){path.push(null);continue}path.push(`${52+i*22.6},${260-Math.min(30,Math.max(0,v))*216/30}`)}let segments=[],segment=[];for(const p of path){if(p)segment.push(p);else if(segment.length){segments.push(segment);segment=[]}}if(segment.length)segments.push(segment);for(const points of segments)svg.append(el('polyline',{points:points.join(' '),fill:'none',stroke:color,'stroke-width':'3'}))}
function draw(){let r=data[Number(query.value)];if(!r)return;
 document.getElementById('status').textContent=`${r.split} · ${r.decision} · ${(r.reasons||[]).join(', ')} · no-hit ${r.counts?.no_hit??'–'}/40`;
 document.getElementById('rgb').src='../../../'+r.image_path;
 document.getElementById('pose').textContent=`GT pose (X,Y,Z): ${r.pose_xyz_m?.map(x=>Number(x).toFixed(2)).join(', ')||'unavailable'} m; logical yaw ${r.logical_yaw_deg}°`;
 const map=document.getElementById('map'),svg=document.getElementById('pose-svg');svg.replaceChildren();
 map.onload=()=>{if(!r.bounds||!r.pose_xyz_m)return;let w=map.naturalWidth,h=map.naturalHeight,b=r.bounds;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);let x=(r.pose_xyz_m[0]-b.x_min)/(b.x_max-b.x_min)*w,z=(r.pose_xyz_m[2]-b.z_min)/(b.z_max-b.z_min)*h;let a=r.logical_yaw_deg*Math.PI/180,L=Math.max(25,Math.min(w,h)*.035);svg.append(el('circle',{cx:x,cy:z,r:Math.max(8,L*.25),fill:'#f08513',stroke:'#fff','stroke-width':'3'}));svg.append(el('line',{x1:x,y1:z,x2:x+Math.cos(a)*L,y2:z+Math.sin(a)*L,stroke:'#f08513','stroke-width':'8','stroke-linecap':'round'}));svg.append(el('circle',{cx:x+Math.cos(a)*L,cy:z+Math.sin(a)*L,r:Math.max(5,L*.16),fill:'#f08513'}))};
 map.src=`../../../2DFP/${r.scene_id}/map.png`;
 const chart=document.getElementById('plot');chart.replaceChildren();chart.append(el('rect',{x:52,y:12,width:882,height:248,fill:'#fff',stroke:'#ccd6e4'}));
 for(let d=0;d<=30;d+=5){let y=260-d*216/30;chart.append(el('line',{x1:52,y1:y,x2:934,y2:y,stroke:'#e5eaf1'}));let t=el('text',{x:38,y:y+5,'text-anchor':'end',fill:'#536781','font-size':14});t.textContent=d;chart.append(t)}
 let axis=el('text',{x:10,y:25,fill:'#536781','font-size':14});axis.textContent='m';chart.append(axis);
 plotCurve(chart,r.stage_depth40_m,'#0064d2');plotCurve(chart,r.target_depth40_m,'#c25a0b');
 for(let i=0;i<40;i++)if(r.stage_depth40_m?.[i]===null)chart.append(el('circle',{cx:52+i*22.6,cy:272,r:2,fill:'#0064d2'}));
}
query.addEventListener('change',draw);query.value='0';draw();
</script></body></html>'''
    return template.replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args()
    hssd = args.hssd_root
    audit = args.audit_root or hssd / "UnLoc" / "geometry_audit_v1"
    rows = [json.loads(line) for file in sorted((audit / "scenes").glob("*.jsonl")) for line in file.open(encoding="utf-8")]
    selected = _select_examples(rows)
    bounds = {}
    for scene in {str(row["scene_id"]) for row in selected}:
        metadata = json.loads((hssd / "2DFP" / scene / "floorplan_metadata.json").read_text())
        bounds[scene] = metadata["world_bounds_m"]
    output = audit / "review" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(selected, bounds), encoding="utf-8")
    print(json.dumps({"path": str(output), "queries": len(selected)}))


if __name__ == "__main__":
    main()
