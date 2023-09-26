from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Pregnancies=int(request.form.get('Pregnancies')),
            Glucose = int(request.form.get('Glucose')),
            BloodPressure = int(request.form.get('BloodPressure')),
            SkinThickness = int(request.form.get('SkinThickness')),
            Insulin = int(request.form.get('Insulin')),
            BMI = float(request.form.get('BMI')),
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
            Age= int(request.form.get('Age'))
        
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)

## http://127.0.0.1:5000/predict to see result 





if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)