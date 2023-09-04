from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/about")
def talha():
    # name = "talha"
    # return render_template('index.html', name2=name)
    return "hellow"


app.run(debug=True)
