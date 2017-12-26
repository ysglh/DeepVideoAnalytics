#!/usr/bin/env python
import logging, time, sys, subprocess


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/fab.log',
                    filemode='a')


if __name__ == '__main__':
    container_type = sys.argv[-1]
    subprocess.check_call(['./copy_defaults.py',])
    subprocess.check_call(['./init_fs.py',])
    block_on_manager = False
    if container_type.strip() == 'worker':
        time.sleep(30)  # To avoid race condition where worker starts before migration is finished
        subprocess.check_call(['./launch_from_env.py','1'])
    subprocess.check_call(['./launch_from_env.py','0'])