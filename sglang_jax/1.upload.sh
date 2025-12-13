#!/bin/bash -x

your_cluster_name=sky-7e59-liuweinan

sky status ${you_cluster_name}

rsync -Pavz * ${your_cluster_name}:/home/gcpuser/sky_workdir

code --remote ssh-remote+${your_cluster_name} "/home/gcpuser/sky_workdir"

