from flask import Flask,render_template

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('sample.html')        #base.html should be in templates folder


if __name__=="__main__":
    app.run(debug=True)