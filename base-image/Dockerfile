from 'statwolf/datascience:3.6'

run apt-get update && apt-get install --yes build-essential

run python3 -m pip install --upgrade pip
run python3 -m pip install jupyter

add config.json /root/.jupyter/jupyter_notebook_config.json

cmd jupyter notebook --allow-root --ip=0.0.0.0 --notebook-dir=/data

run pip install jupyterthemes 
# abilita tema nero

run jt -t onedork -fs 95 -tfs 11 -nfs 115 -cellw 88% -T  

run apt-get update && apt-get install -y --allow-unauthenticated inotify-tools
