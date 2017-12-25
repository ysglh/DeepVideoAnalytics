import requests
import glob
import os
import json
import test_utils


if __name__ == '__main__':
    port = 80
    if not os.path.isfile('creds.json'):
        test_utils.store_token_for_testing()
    token = json.loads(file('creds.json').read())['token']
    headers = {'Authorization': 'Token {}'.format(token)}
    for fname in glob.glob("import_tests/*.json"):
        r = requests.post("http://localhost:{}/api/queries/".format(port), data={'script': file(fname).read()},
                          headers=headers)
        print fname,r.status_code
