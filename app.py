from flask import Flask,request
from prometheus_flask_exporter import PrometheusMetrics
import random

app = Flask(__name__)
metrics = PrometheusMetrics(app, group_by='endpoint')

@app.route('/')
def main():
    return 'OK'

@app.route('/predict')
@metrics.counter('predict', 'Number of prediction',
         labels={'result': lambda: request.result})
def predict():
	request.result = 'invalid'
	number = random.randint(0,1)
	if number == 0 :
	 request.result = 'fraud'
	if number == 1 :
	  request.result = 'not fraud'
	return request.result

if __name__ == '__main__':
    app.run()
