

##########this is old one im gonna use fastapi as new backend so its gonna be main.py file

from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
from markupsafe import Markup
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Helper for line breaks in chat
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))
app.jinja_env.filters['nl2br'] = nl2br

# ✅ Initialize QA chain once at startup
qa_chain = create_qa_chain()
if qa_chain is None:
    raise RuntimeError("❌ QA chain could not be created at startup")

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    error_msg = None

    if request.method == "POST":
        user_input = request.form.get("prompt")
        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})

            try:
                # Use the already initialized QA chain
                result = qa_chain.invoke({"query": user_input})["result"]
                messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_msg = f"Error: {str(e)}"

            session["messages"] = messages
            return render_template("index.html", messages=messages, error=error_msg)

    return render_template("index.html", messages=session.get("messages", []), error=None)

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
