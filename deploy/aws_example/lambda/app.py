import subprocess
from flask import Flask, request, json
from flask import jsonify
from functools import wraps
import boto3
import botocore

app = Flask(__name__)
s3 = boto3.resource('s3')


def get_segment(bname,key):
    try:
        s3.Bucket(bname).download_file(key, '/tmp/temp.mp4')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


@app.route('/')
def process_segment():
    bname = request.form.get('bucket')
    key = request.form.get('key')
    get_segment(bname,key)
    cmd = ['bin/ffmpeg', '-version']
    output = subprocess.check_output(cmd)
    return jsonify({'done':True,'output':output})


if __name__ == '__main__':
    app.run(debug=False)
