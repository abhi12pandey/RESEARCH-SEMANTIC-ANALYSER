from flask import (
    Flask, render_template, request, redirect, make_response, jsonify,
    url_for, flash
)
import os
import re
import jwt
import datetime
from functools import wraps
from werkzeug.utils import secure_filename

# ---- analytics deps ----
import pandas as pd
import numpy as np
import json
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# import your existing DB (do not change)
from db import users_collection, uploads_collection, sessions_collection

# import preprocessing that we updated (must exist)
from preprocessing import process_uploaded_file

# other util imports
import time
import uuid
import traceback

# ---------- CONFIG ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
SECRET_KEY = os.environ.get("SECRET_KEY", "please_change_this_to_a_strong_secret")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY
app.secret_key = SECRET_KEY

# limit file uploads (50 MB) - adjust as needed
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ---------- CONTEXT PROCESSOR ----------
@app.context_processor
def inject_globals():
    try:
        user_email = get_current_user_from_request()
    except Exception:
        user_email = None

    username = None
    if user_email:
        try:
            user_doc = users_collection.find_one({"email": user_email})
            username = user_doc.get("username") if user_doc and user_doc.get("username") else user_email
        except Exception:
            username = user_email

    return {
        "current_year": datetime.datetime.utcnow().year,
        "username": username
    }

# ---------- HELPERS ----------
ALLOWED_EXTENSIONS = {"pdf", "docx", "csv", "xls", "xlsx"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_token(email):
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm="HS256")

def decode_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        return payload.get("email")
    except Exception:
        return None

def get_current_user_from_request():
    token = request.cookies.get("token")
    if not token:
        return None
    return decode_token(token)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        email = get_current_user_from_request()
        if not email:
            flash("Please login to continue.", "error")
            return redirect(url_for("login"))
        return f(email, *args, **kwargs)
    return decorated

def _iso(dt):
    if isinstance(dt, datetime.datetime):
        return dt.replace(microsecond=0).isoformat()
    try:
        return str(dt)
    except Exception:
        return ""

# ---------- ROUTES ----------
@app.route("/")
def index():
    user = get_current_user_from_request()
    return render_template("index.html", username=user)

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if users_collection.find_one({"email": email}):
            flash("User with this email already exists. Please login.", "error")
            return redirect(url_for("signup"))

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password,  # NOTE: plaintext for now - keep as-is to avoid breaking auth
            "created_at": datetime.datetime.utcnow()
        })

        token = generate_token(email)
        resp = make_response(redirect(url_for("dashboard")))
        resp.set_cookie("token", token, httponly=True, samesite="Lax")
        flash("Signup successful. Welcome!", "success")
        return resp

    user = get_current_user_from_request()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("signup.html", username=None)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("login"))

        user = users_collection.find_one({"email": email, "password": password})
        if not user:
            flash("Invalid credentials.", "error")
            return redirect(url_for("login"))

        token = generate_token(email)
        resp = make_response(redirect(url_for("dashboard")))
        resp.set_cookie("token", token, httponly=True, samesite="Lax")

        sessions_collection.insert_one({
            "email": email,
            "login_time": datetime.datetime.utcnow(),
            "ip": request.remote_addr
        })

        flash("Login successful. Welcome back!", "success")
        return resp

    user = get_current_user_from_request()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("login.html", username=None)

@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("index")))
    resp.set_cookie("token", "", expires=0)
    flash("Logged out.", "success")
    return resp

@app.route("/dashboard")
@login_required
def dashboard(current_user_email):
    user = users_collection.find_one({"email": current_user_email})
    username = user["username"] if user else current_user_email
    user_docs = list(uploads_collection.find({"user_email": current_user_email}).sort("ingestion_date", -1))
    return render_template("dashboard.html", username=username, user_docs=user_docs)

@app.route("/upload", methods=["GET","POST"])
@login_required
def upload(current_user_email):
    if request.method == "GET":
        user = users_collection.find_one({"email": current_user_email})
        username = user["username"] if user else current_user_email
        user_docs = list(uploads_collection.find({"user_email": current_user_email}).sort("ingestion_date", -1))
        return render_template("upload.html", username=username, user_docs=user_docs)

    file = request.files.get("document")
    if not file or file.filename == "":
        return jsonify({"status":"error","message":"No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status":"error","message":"Unsupported file type"}), 400

    # create unique filename to avoid collisions
    orig_name = secure_filename(file.filename)
    name_root, ext = os.path.splitext(orig_name)
    unique = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    filename = f"{name_root}_{unique}{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        document_data = process_uploaded_file(save_path, current_user_email)
    except Exception as e:
        # print full traceback for server logs, but return concise message to client
        traceback.print_exc()
        return jsonify({"status":"error","message":f"Preprocessing failed: {str(e)}"}), 500

    # ensure doc_id (generate if process_uploaded_file didn't provide one)
    doc_id_val = document_data.get("doc_id") or f"doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    # prefer saving public file URL (so front-end can link to it) instead of absolute path
    public_file_url = f"/static/uploads/{filename}"

    ingestion_dt = document_data.get("ingestion_date") or datetime.datetime.utcnow()

    doc_record = {
        "user_email": document_data.get("user_email", current_user_email),
        "doc_id": doc_id_val,
        "title": document_data.get("title"),
        "abstract": document_data.get("abstract"),
        "abstractive_summary": document_data.get("abstractive_summary", ""),
        "extractive_summary": document_data.get("extractive_summary", ""),
        "keywords": document_data.get("keywords", []),
        "entities": document_data.get("entities", []),
        "full_text": document_data.get("full_text"),
        "ingestion_date": ingestion_dt,
        "file_path": public_file_url
    }

    uploads_collection.insert_one(doc_record)

    return jsonify({
        "status": "success",
        "message": "Uploaded & ingested successfully!",
        "doc_id": doc_id_val,
        "title": document_data.get("title"),
        "abstract": document_data.get("abstract") or "",
        "summary": document_data.get("abstractive_summary") or document_data.get("extractive_summary") or "",
        "plain_text": document_data.get("full_text") or "",
        "keywords": document_data.get("keywords", []),
        "entities": document_data.get("entities", []),
        "filename": filename,
        "file_path": public_file_url,
        "ingestion_date": _iso(ingestion_dt),
        "redirect_url": url_for("inspect_doc", doc_id=doc_id_val)
    })

@app.route("/inspect/<doc_id>")
@login_required
def inspect_doc(current_user_email, doc_id):
    doc = uploads_collection.find_one({"doc_id": doc_id, "user_email": current_user_email})
    if not doc:
        flash("Document not found or unauthorized.", "error")
        return redirect(url_for("dashboard"))

    return render_template(
        "inspect.html",
        username=current_user_email,
        title=doc.get("title"),
        abstract=doc.get("abstract"),
        summary=doc.get("abstractive_summary") or doc.get("extractive_summary"),
        keywords=doc.get("keywords", []),
        entities=doc.get("entities", []),
        ingestion_date=doc.get("ingestion_date"),
        full_text=(doc.get("full_text", "")[:5000] if isinstance(doc.get("full_text"), str) else doc.get("full_text")),
        doc_id=doc_id   # <-- explicitly pass doc_id to template
    )

@app.route("/visualize/<doc_id>")
@login_required
def visualize_doc(current_user_email, doc_id):
    doc = uploads_collection.find_one({"doc_id": doc_id, "user_email": current_user_email})
    if not doc:
        return jsonify({"status": "error", "message": "Document not found"}), 404

    full_text = doc.get("full_text") or ""
    summary = doc.get("abstractive_summary") or doc.get("extractive_summary") or ""
    entities = doc.get("entities") or []
    keywords = doc.get("keywords") or []

    # Entities DF (string/dict safe)
    ent_rows = []
    for e in entities:
        if isinstance(e, str):
            ent_rows.append({"text": e, "label": "OTHER"})
        elif isinstance(e, dict):
            ent_rows.append({
                "text": e.get("text") or e.get("entity") or "",
                "label": e.get("label") or e.get("type") or "OTHER"
            })
    ent_df = pd.DataFrame(ent_rows) if ent_rows else pd.DataFrame(columns=["text","label"])

    # Paragraphs DF
    if isinstance(full_text, list):
        paras = [(p if isinstance(p, str) else (p.get("cleaned_text") or p.get("raw_text") or "")) for p in full_text]
    else:
        paras = [p.strip() for p in (full_text or "").split("\n\n") if p.strip()]
    para_df = pd.DataFrame({"paragraph": paras})
    if not para_df.empty:
        para_df["length"] = para_df["paragraph"].apply(lambda s: len(s.split()))

    # Word frequency (basic stopwords)
    text_for_freq = full_text if isinstance(full_text, str) else " ".join(paras)
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-']+", text_for_freq)]
    stop = set("""a an the and or if but to for with on in of from by as at is am are was were be been being it this that these those you your we our they their he she his her its not no do does did so than then too very can could should would will may might must have has had i me my""".split())
    toks = [t for t in tokens if t not in stop and len(t) >= 3]
    if toks:
        wf = pd.Series(toks).value_counts().reset_index()
        wf.columns = ["word", "count"]
        wf_top = wf.head(25)
    else:
        wf_top = pd.DataFrame(columns=["word","count"])

    # Build figures
    figs = {}
    if not ent_df.empty:
        lab = (ent_df.assign(label=ent_df["label"].fillna("OTHER"))
                      .groupby("label").size().reset_index(name="count")
                      .sort_values("count", ascending=False))
        figs["entities_bar"] = px.bar(lab, x="label", y="count", title="Entities by Label")

    if not para_df.empty:
        figs["para_len_hist"] = px.histogram(para_df, x="length", nbins=30, title="Paragraph Length (word count)")

    if not wf_top.empty:
        figs["word_freq_bar"] = px.bar(
            wf_top.sort_values("count"),
            x="count", y="word", orientation="h",
            title="Top Word Frequencies (stopwords removed)"
        )

    full_wc = len(text_for_freq.split()) if text_for_freq else 0
    sum_wc = len((summary or "").split())
    svf = pd.DataFrame({"part": ["Summary","Full Text"], "words": [sum_wc, full_wc]})
    figs["summary_vs_full"] = px.bar(svf, x="part", y="words", title="Summary vs Full Text — Word Count")

    figs_json = {k: json.dumps(v, cls=PlotlyJSONEncoder) for k, v in figs.items()}

    # notes = {
    #     # "entities_bar": "NER labels ka distribution (PERSON/ORG/LOC...). Document kin types par focus karta hai, yeh dikhata hai.",
    #     # "para_len_hist": "Paragraph length (words) ka histogram — structure/readability ka quick view.",
    #     # "word_freq_bar": "Stopwords hata kar top repeated words — document ke main concepts ko highlight karta hai.",
    #     # "summary_vs_full": "Summary aur Full Text ke word counts — compression/coverage samajhne ke liye."
    # }

    return jsonify({"status": "ok", "figs": figs_json,  "keywords": keywords})

# ---------- AGGREGATE HELPERS ----------
def _docs_to_dataframe(docs):
    rows = []
    for d in docs:
        full_text = d.get("full_text") or ""
        summary = d.get("abstractive_summary") or d.get("extractive_summary") or ""
        keywords = d.get("keywords") or []
        entities = d.get("entities") or []
        tokens = len(full_text.split()) if isinstance(full_text, str) else None
        summary_len = len(summary.split()) if isinstance(summary, str) else None
        entities_count = len(entities) if isinstance(entities, (list, tuple)) else None
        fp = (d.get("file_path") or "").lower()
        source = "PDF" if fp.endswith(".pdf") else ("DOCX" if fp.endswith(".docx") else "Upload")
        topic = (keywords[0] if isinstance(keywords, list) and keywords else None)
        rows.append({
            "doc_id": d.get("doc_id"),
            "title": d.get("title"),
            "source": source,
            "date": d.get("ingestion_date"),
            "tokens": tokens,
            "summary_len": summary_len,
            "topic": topic,
            "entities_count": entities_count,
            "keywords": "; ".join(keywords) if isinstance(keywords, list) else str(keywords)
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["doc_id","title","source","date","tokens","summary_len","topic","entities_count","keywords"]
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["tokens","summary_len","entities_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@app.route("/analytics")
@login_required
def analytics(current_user_email):
    docs = list(uploads_collection.find({"user_email": current_user_email}).sort("ingestion_date", -1))
    df = _docs_to_dataframe(docs)
    kpis = {
        "docs": int(df.shape[0]),
        "median_tokens": int(np.nanmedian(df["tokens"])) if "tokens" in df and not df["tokens"].isna().all() else None,
        "avg_summary": float(np.nanmean(df["summary_len"])) if "summary_len" in df and not df["summary_len"].isna().all() else None,
        "mean_sentiment": None,
        "avg_readability": None,
    }
    figs = {}
    if "date" in df and df["date"].notna().any():
        ts = (df.dropna(subset=["date"]).assign(day=lambda d: d["date"].dt.to_period("D").dt.to_timestamp())
                .groupby("day").agg(docs=("doc_id","count")).reset_index())
        figs["docs_over_time"] = px.line(ts, x="day", y="docs", title="Documents Over Time")
    if "topic" in df and df["topic"].notna().any():
        topic = (df.assign(topic=df["topic"].fillna("Unlabeled"))
                   .groupby("topic").agg(count=("doc_id","count")).reset_index()
                   .sort_values("count", ascending=False))
        figs["topics_count"] = px.bar(topic, x="topic", y="count", title="Top Topics by Document Count")
    if "source" in df and not df.empty:
        src = df.groupby("source").agg(count=("doc_id","count")).reset_index()
        figs["source_pie"] = px.pie(src, names="source", values="count", hole=0.35, title="Documents by Source")
    if {"summary_len","tokens"}.issubset(df.columns) and not df.empty:
        figs["len_vs_tokens"] = px.scatter(df, x="tokens", y="summary_len",
                                           hover_data=["doc_id","title","topic","source"],
                                           title="Summary Length vs Tokens")
    if {"entities_count","tokens"}.issubset(df.columns) and not df.empty:
        figs["entities_vs_tokens"] = px.scatter(df, x="tokens", y="entities_count",
                                                hover_data=["doc_id","title","topic","source"],
                                                title="Entities vs Tokens")
    figs_json = {k: json.dumps(v, cls=PlotlyJSONEncoder) for k, v in figs.items()}
    table_cols = df.columns.tolist()
    table_rows = df.head(50).to_dict(orient="records")
    user = users_collection.find_one({"email": current_user_email})
    username = user["username"] if user else current_user_email
    return render_template("analytics.html",
                           username=username, kpis=kpis,
                           figs_json=figs_json, table_cols=table_cols, table_rows=table_rows)

@app.route("/analytics.csv")
@login_required
def analytics_csv(current_user_email):
    docs = list(uploads_collection.find({"user_email": current_user_email}))
    df = _docs_to_dataframe(docs)
    csv_data = df.to_csv(index=False)
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=analytics.csv"
    return resp

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
