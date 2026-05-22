from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

# ---------------------- LOAD CSV ----------------------
human_qrels = pd.read_csv(
    os.path.expanduser("qrels.test.mrtydi.txt"),
    sep="\s+",
    header=None,
    names=["qid", "Q0", "docid", "rank"]
)

columns = ["qid", "Q0", "docid", "rank", "score", "querytext", "doctext"]

pairs = pd.read_csv(
    "3cross_endocder_rrf.txt",
    sep="\t",
    header=None,
    names=columns
)

# Optional: create a query dictionary if querytext may be missing
queries = {row["qid"]: row["querytext"] for _, row in pairs.iterrows() if pd.notna(row["querytext"])}

# ---------------------- SETUP RESULTS FILE ----------------------

OUTPUT_FILE = "syntheticjudgmentsmrtydi.csv"

if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["query_id", "doc_id", "relevant"]).to_csv(OUTPUT_FILE, index=False)

def get_completed_queries():
    df = pd.read_csv(OUTPUT_FILE)
    return set(df["query_id"].unique())

def get_next_query():
    completed = get_completed_queries()
    for qid in pairs["qid"].unique():
        if qid not in completed:
            return qid
    return None

# ---------------------- HTML TEMPLATE ----------------------
TEMPLATE = """
<html>
<head>
<meta charset="UTF-8">
<title>Annotate Query {{ qid }}</title>
<style>
body { font-family: Arial; margin: 40px; }
.box { border: 1px solid #ccc; padding: 20px; margin-bottom: 40px; border-radius: 10px; }
.query { font-weight: bold; font-size: 20px; margin-bottom: 15px; }
.doc { padding: 10px; margin: 10px 0; border-radius: 6px; }
button { padding: 10px 20px; font-size: 16px; }
.progress { margin-bottom:20px; }
.progress-bar {
  width: {{ progress }}%;
  height: 20px;
  background-color: #4CAF50;
}
.progress-container {
  width: 100%;
  background-color: #ddd;
}
</style>
</head>

<body>
<h1>Human Annotation Tool</h1>

<div class="progress">
  <div>Total: {{ total }} queries — Completed: {{ done }} — Remaining: {{ total - done }}</div>
  <div class="progress-container">
    <div class="progress-bar"></div>
  </div>
</div>

<div class="box">
  <form method="post">

    <div class="query">
        <b>Query {{ qid }}:</b><br>{{ query_text }}
    </div>

    {% for did, text, is_relevant in docs %}
    <div class="doc" style="background-color: {% if is_relevant %}#ffcccc{% else %}#f7f7f7{% endif %};">
      <input type="checkbox" name="{{ did }}">
      <b>Document {{ did }}</b><br>
      {{ text }}
    </div>
    {% endfor %}

    <button type="submit">Save & Next</button>
  </form>
</div>

</body>
</html>
"""

# ---------------------- ROUTES ----------------------
@app.route("/", methods=["GET"])
def index():
    next_q = get_next_query()
    if next_q is None:
        return "<h1>All queries annotated ✔</h1>"
    return redirect(url_for("annotate", qid=next_q))

@app.route("/annotate/<qid>", methods=["GET", "POST"])
def annotate(qid):
    df_judgments = pd.read_csv(OUTPUT_FILE)

    if request.method == "POST":
        # Remove old judgments for this query (restart-safe)
        df_judgments = df_judgments[df_judgments["query_id"] != int(qid)]

        group = pairs[pairs["qid"] == int(qid)].sort_values("score", ascending=False).head(10)

        for _, row in group.iterrows():
            did = row["docid"]
            relevant = 1 if did in request.form else 0
            df_judgments.loc[len(df_judgments)] = [qid, did, relevant]

        df_judgments.to_csv(OUTPUT_FILE, index=False)

        next_q = get_next_query()
        if next_q is None:
            return "<h1>All queries annotated ✔</h1>"
        return redirect(url_for("annotate", qid=next_q))

    # GET request
    group = pairs[pairs["qid"] == int(qid)].sort_values("score", ascending=False).head(10)
    # Get relevant docids for this query from human_qrels
    relevant_docids = set(human_qrels[human_qrels["qid"] == int(qid)]["docid"])
    docs = [(row["docid"], row["doctext"], row["docid"] in relevant_docids) for _, row in group.iterrows()]

    # Use query text from CSV if available, otherwise fallback
    query_text = group["querytext"].iloc[0] if not group.empty and pd.notna(group["querytext"].iloc[0]) else queries.get(int(qid), "[Query not found]")

    done = len(get_completed_queries())
    total = pairs["qid"].nunique()
    progress = (done / total) * 100 if total > 0 else 0

    return render_template_string(
        TEMPLATE,
        qid=qid,
        query_text=query_text,
        docs=docs,
        done=done,
        total=total,
        progress=progress
    )

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
