from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')



## http://127.0.0.1:5000/predict to see result 





if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)