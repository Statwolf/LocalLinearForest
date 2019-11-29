from llf/python:3.6

run apt-get update && apt-get install -y --allow-unauthenticated inotify-tools

add deps.txt /deps.txt
run pip install -r /deps.txt