#!/usr/bin/env python
import django
import sys, os, logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/flask.log',
                    filemode='a')
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.models import TEvent
from django.conf import settings
from dvaapp.task_handlers import handle_perform_analysis, handle_perform_indexing, handle_perform_detection


from flask import Flask

app = Flask(__name__)


@app.route('/<pk>/')
def process_task(pk):
    start = TEvent.objects.get(pk=pk)
    logging.info("Executing {} {}".format(start.operation,pk))
    if start.operation == 'perform_indexing':
        handle_perform_indexing(start)
    elif start.operation == 'perform_detection':
        handle_perform_detection(start)
    elif start.operation == 'perform_analysis':
        handle_perform_analysis(start)
    else:
        raise ValueError("Unknown task name {}".format(start.operation))
    return "Done"


@app.route('/')
def ready():
    return "running"


if __name__ == '__main__':
    with open('flask.pid','w') as pidfile:
        pidfile.write(str(os.getpid()))
    app.run(port=settings.GLOBAL_MODEL_FLASK_SERVER_PORT,debug=False)