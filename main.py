import os
import cv2
import cnn_model as cm
from flask import Flask, jsonify, render_template, request

__author__ = 'yungfengchang'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Flask Webapp
@app.route('/')
def index():
	return render_template("upload.html")

# Post request
@app.route("/upload",methods=['POST']) 
def upload():
	if request.method == 'POST':
		# Takes an mnist image as a post request
		target = os.path.join(APP_ROOT,'Database/')

		if not os.path.isdir(target):
			os.mkdir(target)

		for file in request.files.getlist("file"):
			filename = file.filename
			destination = "/".join([target,filename])
			file.save(destination)

		input_image_old= cv2.imread(destination,0)

		if input_image_old.shape[0] == 28 and input_image_old.shape[1] == 28:

			output = cm.convolutional(destination)
			list = [{'The classification is : ': str(output)}]
			# Returns a JSON blob with the classification of that image
			return jsonify(results=list)

		else:
			# Invalid Input should return 404
			response = jsonify({'code': 404,'message': 'No interface defined for URL'})
			response.status_code = 404
			return response

	else:
		return redirect('/')		


if __name__ == "__main__":
    app.run(port=4555,debug=True)