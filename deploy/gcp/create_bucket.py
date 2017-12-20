from config import mediabucket, region
import os
os.system('gsutil mb -c regional -l {} gs://{}'.format(region,mediabucket))
os.system('gsutil iam ch allUsers:objectViewer gs://{}'.format(mediabucket))