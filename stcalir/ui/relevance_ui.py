from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template_string, request, redirect, url_for

from stcalir.utils import get_logger

logger = get_logger(__name__)

RELEVANCE_TEMPLATE = """
<!DOCTYPE html>
<html lang="{{ lang }}">
<head>
  <meta charset="UTF-8">
  <title>Relevance Annotation — Topic {{ qid }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; direction: {{ direction }}; }
    h2 { color: #333; }
    .progress-bar-outer { width: 100%; background: #eee; border-radius: 8px; margin-bottom: 20px; }
    .progress-bar-inner { height: 18px; background: #2196f3; border-radius: 8px;
                          width: {{ pct }}%; transition: width .3s; }
    .progress-label { font-size: 13px; color: #555; margin-bottom: 16px; }
    .query-box { background: #e3f2fd; border-radius: 8px; padding: 16px 20px;
                 font-size: 18px; font-weight: bold; margin-bottom: 24px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 18px 20px;
            margin-bottom: 20px; background: #fafafa; }
    .card-header { font-size: 12px; color: #888; margin-bottom: 8px; }
    .card-text { font-size: 15px; line-height: 1.7; margin-bottom: 14px; }
    .radio-group label { margin-right: 24px; font-size: 15px; cursor: pointer; }
    .radio-group input { margin-right: 6px; }
    .btn { padding: 12px 32px; font-size: 16px; background: #4caf50; color: white;
           border: none; border-radius: 8px; cursor: pointer; margin-top: 20px; }
    .btn:hover { background: #388e3c; }
    .done { text-align: center; padding: 80px; font-size: 22px; color: #333; }
    .rank-badge { display: inline-block; background: #2196f3; color: white;
                  border-radius: 50%; width: 26px; height: 26px; text-align: center;
                  line-height: 26px; font-size: 13px; margin-right: 8px; }
  </style>
</head>
<body>
  {% if done %}
    <div class="done">
      <h2>✅ Annotation Complete!</h2>
      <p>All {{ total }} topics annotated.</p>
      <p>Qrels saved to <code>{{ output_path }}</code></p>
      <p>You can now close this tab and continue in the notebook.</p>
    </div>
  {% else %}
    <h2>Relevance Annotation</h2>
    <div class="progress-bar-outer">
      <div class="progress-bar-inner"></div>
    </div>
    <div class="progress-label">Topic {{ current }} of {{ total }} — {{ pct }}% complete</div>

    <div class="query-box">🔍 {{ query }}</div>

    <form method="POST">
      <input type="hidden" name="qid" value="{{ qid }}">
      {% for doc in docs %}
      <div class="card">
        <div class="card-header">
          <span class="rank-badge">{{ doc.rank }}</span>
          Passage ID: {{ doc.pid }}
        </div>
        <div class="card-text">{{ doc.text }}</div>
        <div class="radio-group">
          <label><input type="radio" name="rel_{{ doc.pid }}" value="1" required> Relevant</label>
          <label><input type="radio" name="rel_{{ doc.pid }}" value="0"> Not Relevant</label>
        </div>
      </div>
      {% endfor %}
      <button type="submit" class="btn">Save &amp; Next Topic →</button>
    </form>
  {% endif %}
</body>
</html>
"""


def run_relevance_ui(
    top10_path: str,            # JSONL: {"qid": str, "query": str, "candidates": [{"pid", "text", "rank"}, ...]}
    output_path: str,           # TREC qrels output
    host: str = "0.0.0.0",
    port: int = 5000,
    lang: str = "arabic",
) -> None:
    """
    Flask UI for human relevance annotation.
    Shows the query + top-10 passages; annotator marks each as relevant (1) or not (0).
    """
    top10_path  = Path(top10_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    direction   = "rtl" if lang == "arabic" else "ltr"

    # Load all topics + candidates
    topics_data: list[dict] = []
    with open(top10_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                topics_data.append(json.loads(line))

    topic_map = {str(t["qid"]): t for t in topics_data}

    def _completed() -> set[str]:
        done: set[str] = set()
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        done.add(parts[0])
        return done

    def _next_topic():
        done = _completed()
        for t in topics_data:
            if str(t["qid"]) not in done:
                return t
        return None

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            qid = request.form.get("qid", "")
            topic = topic_map.get(qid, {})
            with open(output_path, "a", encoding="utf-8") as f:
                for doc in topic.get("candidates", []):
                    pid = doc["pid"]
                    rel = int(request.form.get(f"rel_{pid}", 0))
                    f.write(f"{qid}\t0\t{pid}\t{rel}\n")
            return redirect(url_for("index"))

        topic = _next_topic()
        done  = len(_completed())
        total = len(topics_data)

        if topic is None:
            return render_template_string(
                RELEVANCE_TEMPLATE, done=True, total=total, output_path=str(output_path),
                lang=lang, direction=direction, current=done, qid="", query="", docs=[], pct=100
            )

        pct  = int(done / total * 100)
        docs = [
            {"pid": c["pid"], "text": c.get("text", ""), "rank": i + 1}
            for i, c in enumerate(topic.get("candidates", []))
        ]
        return render_template_string(
            RELEVANCE_TEMPLATE, done=False, current=done + 1, total=total,
            qid=topic["qid"], query=topic.get("query", ""), docs=docs,
            lang=lang, direction=direction, pct=pct, output_path=str(output_path)
        )

    logger.info(f"Relevance UI → http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def prepare_top10_for_ui(
    top10_run: dict[str, list[tuple[str, float]]],
    topics: list[dict],
    passage_lookup: dict[str, str],
    output_path: str,
) -> None:
    """
    Serialize the top-10 run + passage texts into a JSONL file ready for the UI.
    """
    topic_map = {str(t["qid"]): t["text"] for t in topics}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for qid, ranked in top10_run.items():
            candidates = [
                {"pid": pid, "text": passage_lookup.get(pid, ""), "rank": rank + 1}
                for rank, (pid, _) in enumerate(ranked)
            ]
            rec = {
                "qid":        qid,
                "query":      topic_map.get(qid, ""),
                "candidates": candidates,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Top-10 UI data saved → {output_path}")
