from config import mediabucket
import os
os.system('gsutil mb gs://{}'.format(mediabucket))
os.system('gsutil iam ch allUsers:objectViewer gs://{}'.format(mediabucket))