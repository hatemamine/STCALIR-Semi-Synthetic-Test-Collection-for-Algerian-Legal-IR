from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from flask import Flask, render_template_string, request, redirect, url_for

from stcalir.utils import get_logger

logger = get_logger(__name__)

TOPIC_TEMPLATE = """
<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
  <meta charset="UTF-8">
  <title>Topic Creation — {{ current }}/{{ total }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; direction: {{ direction }}; }
    .progress-bar-outer { width: 100%; background: #eee; border-radius: 8px; margin-bottom: 20px; }
    .progress-bar-inner { height: 18px; background: #4caf50; border-radius: 8px;
                          width: {{ pct }}%; transition: width .3s; }
    .passage-box { background: #f8f8f8; border: 1px solid #ccc; border-radius: 8px;
                   padding: 20px; margin-bottom: 25px; font-size: 15px; line-height: 1.7; }
    .pid-label { font-size: 11px; color: #888; margin-bottom: 8px; }
    textarea { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #bbb;
               border-radius: 6px; resize: vertical; }
    .btn { padding: 10px 28px; font-size: 15px; background: #2196f3; color: white;
           border: none; border-radius: 6px; cursor: pointer; margin-top: 12px; }
    .btn:hover { background: #1976d2; }
    .btn-skip { background: #9e9e9e; margin-left: 12px; }
    .done { text-align: center; padding: 60px; font-size: 22px; color: #333; }
  </style>
</head>
<body>
  {% if done %}
    <div class="done">
      <h2>✅ All topics created!</h2>
      <p>{{ total }} topics saved to <code>{{ output_path }}</code></p>
      <p>You can now close this tab and continue in the notebook.</p>
    </div>
  {% else %}
    <h2>Topic Creation — Passage {{ current }} of {{ total }}</h2>
    <div class="progress-bar-outer">
      <div class="progress-bar-inner"></div>
    </div>
    <p>Read the passage below and write a natural search query a user might type to find it.</p>
    <div class="passage-box">
      <div class="pid-label">Passage ID: {{ pid }}</div>
      {{ passage }}
    </div>
    <form method="POST">
      <input type="hidden" name="pid" value="{{ pid }}">
      <textarea name="query" rows="3" placeholder="Write your query here...">{{ prefill }}</textarea><br>
      <button type="submit" class="btn">Save &amp; Next →</button>
      <button type="submit" name="skip" value="1" class="btn btn-skip">Skip</button>
    </form>
  {% endif %}
</body>
</html>
"""


def run_topic_ui(
    seeds_path: str,
    output_path: str,
    host: str = "0.0.0.0",
    port: int = 5001,
    lang: str = "arabic",
) -> None:
    """
    Launch a Flask web app for human topic creation.
    Each page shows one passage seed; the annotator writes a query.
    Topics are saved incrementally to output_path (JSONL).
    """
    seeds = []
    with open(seeds_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    direction = "rtl" if lang == "arabic" else "ltr"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed pids
    def _completed() -> set:
        done = set()
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        done.add(rec.get("source_pid", ""))
        return done

    def _next_seed():
        done = _completed()
        for s in seeds:
            if s["pid"] not in done:
                return s
        return None

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            pid   = request.form.get("pid", "")
            query = request.form.get("query", "").strip()
            skip  = request.form.get("skip", "")
            if not skip and query:
                qid = str(len(_completed()))
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"qid": qid, "text": query, "source_pid": pid}, ensure_ascii=False) + "\n")
            return redirect(url_for("index"))

        seed = _next_seed()
        done  = len(_completed())
        total = len(seeds)

        if seed is None:
            return render_template_string(
                TOPIC_TEMPLATE, done=True, total=total, output_path=str(output_path),
                lang=lang, direction=direction, current=done, pid="", passage="", prefill="", pct=100
            )

        pct = int(done / total * 100)
        return render_template_string(
            TOPIC_TEMPLATE, done=False, current=done + 1, total=total,
            pid=seed["pid"], passage=seed["text"], prefill="",
            lang=lang, direction=direction, pct=pct, output_path=str(output_path)
        )

    logger.info(f"Topic UI → http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
