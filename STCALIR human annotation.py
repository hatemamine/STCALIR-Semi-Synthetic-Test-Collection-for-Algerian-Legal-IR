from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import os
from collections import defaultdict
import glob
# topic 19, 37, 42, 49, 59, 80, 92,  no relevent doc 
# topic 30
app = Flask(__name__)
app.secret_key = "supersecretkey"

def load_ranking_file(path):
    """Load a ranking file of format: qid, docid, rank."""
    return pd.read_csv(path, sep="\t", names=["qid", "docid", "rank", "score"])

def rrf(ranking_lists, k=60):
    """Apply RRF fusion to a list of ranking lists (docid lists)."""
    scores = defaultdict(float)
    for ranking in ranking_lists:
        for rank, docid in enumerate(ranking):
            scores[docid] += 1.0 / (k + rank + 1)
    # Sort documents by RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def rrf_fusion(run_files, output_path=None, k=60, top_k=1000):
    """
    Apply RRF to multiple ranking files with identical structure.
    
    run_files: list of file paths
    output_path: where to save fused run (optional)
    Returns: fused DataFrame
    """
    
    # 1) Load all runs
    runs = [load_ranking_file(f) for f in run_files]

    # All topics from first file
    topics = runs[0]["qid"].unique()

    fused_rows = []

    for qid in topics:
        ranking_lists = []

        # 2) Extract docids per topic from each run
        for df in runs:
            docs = df[df["qid"] == qid].sort_values("rank")["docid"].tolist()
            ranking_lists.append(docs)

        # 3) Apply RRF
        fused_docs = rrf(ranking_lists, k=k)

        # 4) Keep top_k
        for rank, (docid, score) in enumerate(fused_docs[:top_k], start=1):
            fused_rows.append([qid, docid, rank])

    # Create DataFrame
    fused_df = pd.DataFrame(fused_rows, columns=["qid", "docid", "rank"])

    # Optional save
    if output_path:
        fused_df.to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Saved RRF run to: {output_path}")

    return fused_df


# ---------------------- LOAD CSVs ----------------------

ss = ["./secondstage/"+f for f in os.listdir("./secondstage/")]
print(ss)
print(len(ss))

rrf_fusion(ss, output_path="3cross_endocder_rrf.txt")



# ---------------------- LOAD CSV ----------------------
# Load corpus
df = pd.read_csv("Corpus Algerian Legal Texts.csv")
df = df[['passage_id', 'text_with_summary']]
df = df.rename(columns={'passage_id': 'id', 'text_with_summary': 'text'})
df["id"] = df["id"].replace(' ', '_', regex=True)

# Load topics
df_topic = pd.read_csv("Topics Algerian Legal Texts.csv")
df_topic = df_topic.reset_index()[["topic_id", "topic_title"]]
df_topic = df_topic.rename(columns={'topic_id': 'id', 'topic_title': 'text'})
df_topic.to_csv('STCALIR-Topics.tsv', sep="\t", header=None, index=False)

# Load pairs
columns = ["qid", "docid", "rank"]
pairs = pd.read_csv("3cross_endocder_rrf.txt", sep="\t", header=None, names=columns)

# Merge query text
pairs = pairs.merge(df_topic.rename(columns={'id':'qid', 'text':'querytext'}), on='qid', how='left')
# Merge document text
pairs = pairs.merge(df.rename(columns={'id':'docid', 'text':'doctext'}), on='docid', how='left')

# ---------------------- RESULTS FILE ----------------------
os.makedirs("results", exist_ok=True)
OUTPUT_FILE = "STCALIRjudgments.csv"
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["query_id", "doc_id", "relevant"]).to_csv(OUTPUT_FILE, index=False)

# ---------------------- HELPERS ----------------------
def get_completed_queries():
    df_j = pd.read_csv(OUTPUT_FILE)
    return set(df_j["query_id"].unique())

def get_query_list():
    return sorted(pairs["qid"].unique())

def get_next_query(qid):
    q_list = get_query_list()
    idx = q_list.index(int(qid))
    if idx + 1 < len(q_list):
        return q_list[idx + 1]
    return None

def get_previous_query(qid):
    q_list = get_query_list()
    idx = q_list.index(int(qid))
    if idx - 1 >= 0:
        return q_list[idx - 1]
    return None

# ---------------------- TEMPLATE ----------------------
TEMPLATE = """
<html>
<head>
<meta charset="UTF-8">
<title>أداة التقييم البشري</title>
<style>
body { font-family: Arial; margin: 40px; direction: rtl; text-align: right; }
.box { border: 1px solid #ccc; padding: 20px; margin-bottom: 40px; border-radius: 10px; }
.doc { padding: 10px; margin: 10px 0; background: #f7f7f7; border-radius: 6px; transition: background 0.3s; }
.doc.checked { background-color: #d4f7d4; }  /* light green for checked docs */
button { padding: 10px 20px; font-size: 16px; margin-left: 10px; }
.progress { margin-bottom:20px; }
.progress-bar { width: {{ progress }}%; height: 20px; background-color: #4CAF50; }
.progress-container { width: 100%; background-color: #ddd; }
.nav-buttons { margin-bottom: 20px; }
input[type="checkbox"] { transform: scale(1.2); margin-left:10px; }
</style>
</head>
<body>
<h1>أداة التقييم البشري</h1>

<div class="progress">
  <div>إجمالي الاستعلامات: {{ total }} — المكتملة: {{ done }} — المتبقية: {{ total - done }}</div>
  <div class="progress-container">
    <div class="progress-bar"></div>
  </div>
</div>

<div class="nav-buttons">
    {% if prev_q %}
        <a href="{{ url_for('annotate', qid=prev_q) }}"><button type="button">⬅ السابق</button></a>
    {% else %}
        <button type="button" disabled style="opacity:0.5; cursor:not-allowed;">⬅ السابق</button>
    {% endif %}
    {% if next_q %}
        <a href="{{ url_for('annotate', qid=next_q) }}"><button type="button">التالي ➡</button></a>
    {% else %}
        <button type="button" disabled style="opacity:0.5; cursor:not-allowed;">التالي ➡</button>
    {% endif %}
</div>

<div class="box">
  <form method="post">
    <button type="submit" name="action" value="save_next">💾 حفظ والانتقال</button>

    <div class="query">
        <b>سؤال {{ qid }}:</b><br>{{ query_text }}
    </div>

    {% for did, text in docs %}
    <div class="doc {% if did in selected_docs %}checked{% endif %}">
      <input type="checkbox" name="{{ did }}" {% if did in selected_docs %}checked{% endif %}>
      <b>المستند {{ did }}</b><br>
      {{ text }}
    </div>
    {% endfor %}

    <button type="submit" name="action" value="save_next">💾 حفظ والانتقال</button>
  </form>
</div>

<script>
document.querySelectorAll('.doc input[type="checkbox"]').forEach(function(cb){
  cb.addEventListener('change', function(){
    if(cb.checked){ cb.parentElement.classList.add('checked'); }
    else { cb.parentElement.classList.remove('checked'); }
  });
});
</script>

</body>
</html>
"""

# ---------------------- ROUTES ----------------------
@app.route("/")
def index():
    qid = get_query_list()[0]
    return redirect(url_for("annotate", qid=qid))

@app.route("/annotate/<qid>", methods=["GET", "POST"])
def annotate(qid):
    qid = int(qid)
    df_judgments = pd.read_csv(OUTPUT_FILE)

    if request.method == "POST":
        df_judgments = df_judgments[df_judgments["query_id"] != qid]
        group = pairs[pairs["qid"] == qid].sort_values("rank", ascending=True).head(10)
        for _, row in group.iterrows():
            did = row["docid"]
            relevant = 1 if did in request.form else 0
            df_judgments.loc[len(df_judgments)] = [qid, did, relevant]
        df_judgments.to_csv(OUTPUT_FILE, index=False)

        next_q = get_next_query(qid)
        if next_q is None:
            return "<h1>تم تقييم جميع الاستعلامات ✔</h1>"
        return redirect(url_for("annotate", qid=next_q))

    # GET
    group = pairs[pairs["qid"] == qid].sort_values("rank", ascending=True).head(10)
    docs = [(row["docid"], row["doctext"]) for _, row in group.iterrows()]
    old = df_judgments[df_judgments["query_id"] == qid]
    selected_docs = set(old[old["relevant"] == 1]["doc_id"].tolist())
    query_text = group["querytext"].iloc[0] if not group.empty else "[استعلام مفقود]"

    prev_q = get_previous_query(qid)
    next_q = get_next_query(qid)
    done = len(get_completed_queries())
    total = pairs["qid"].nunique()
    progress = (done / total) * 100 if total else 0

    return render_template_string(
        TEMPLATE,
        qid=qid,
        docs=docs,
        query_text=query_text,
        selected_docs=selected_docs,
        prev_q=prev_q,
        next_q=next_q,
        done=done,
        total=total,
        progress=progress
    )

# ----------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
