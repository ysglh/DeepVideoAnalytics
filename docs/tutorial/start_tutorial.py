#!/usr/bin/env python
import os, requests, subprocess, time, webbrowser

if __name__ == '__main__':
    print "Checking if docker is running"
    max_minutes = 20
    try:
        subprocess.check_call(["docker",'ps','-a'])
    except:
        raise SystemError("Docker is not running")
    try:
        subprocess.check_call(["docker-compose",'ps'],
                              cwd=os.path.join(os.path.dirname(__file__),'../../deploy/cpu'))
    except:
        raise SystemError("Docker-compose is not available")
    print "Trying to launch containers, first time it might take a while to download container images"
    try:
        compose_process = subprocess.Popen(["docker-compose",'up','-d'],
                                           cwd=os.path.join(os.path.dirname(__file__),'../../deploy/cpu'))
    except:
        raise SystemError("Could not start container")
    while max_minutes:
        print "Waiting for {max_minutes} minutes, while checking if DVA is running".format(max_minutes=max_minutes)
        try:
            r = requests.get("http://localhost:8000")
            if r.ok:
                print "Open browser window and go to http://localhost:8000 to access DVA Web UI"
                print 'Use following auth code to use jupyter notebook on  '
                print subprocess.check_output(["jupyter",'notebook','list'])
                print 'For windows you might need to replace "localhost" with ip address of docker-machine '
                webbrowser.open("http://localhost:8000")
                webbrowser.open("http://localhost:8888")
                break
        except:
            pass
        time.sleep(60)
        max_minutes -= 1
    compose_process.wait()