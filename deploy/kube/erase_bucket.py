from config import mediabucket
import os
os.system('gsutil -m rm gs://{}/**'.format(mediabucket))
