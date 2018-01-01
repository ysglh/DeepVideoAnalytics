from config import mediabucket, region, cors_origin
import os, json
os.system('gsutil mb -c regional -l {} gs://{}'.format(region,mediabucket))
os.system('gsutil iam ch allUsers:objectViewer gs://{}'.format(mediabucket))
with open('cors.json','w') as out:
    json.dump([
    {
      "origin": [cors_origin],
      "responseHeader": ["Content-Type"],
      "method": ["GET", "HEAD"],
      "maxAgeSeconds": 3600
    }
    ],out)
os.system('gsutil cors set cors.json gs://{}'.format(mediabucket))