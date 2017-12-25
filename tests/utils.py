import shutil
import subprocess


def store_token_for_testing():
    subprocess.check_output(['python','generate_testing_token.py'],cwd="../server/scripts")
    shutil.move('../server/scripts/creds.json','creds.json')


def migrate():
    subprocess.check_output(['./migrate.sh'],cwd="../server/")
