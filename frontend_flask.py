from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/post_image", methods=["POST"])
def post_image():
    return "Image received"


if __name__ == "__main__":
    app.run(debug=True)
