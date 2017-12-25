#!/usr/bin/env python
import requests
import os
import json
import utils

if __name__ == '__main__':
    port = 80
    if not os.path.isfile('creds.json'):
        utils.store_token_for_testing()
    token = json.loads(file('creds.json').read())['token']
    headers = {'Authorization': 'Token {}'.format(token)}
    r = requests.post("http://localhost:{}/api/queries/".format(port),
                      data={'script': file('dvaapp/test_scripts/url.json').read()},
                      headers=headers)
    print r.status_code
