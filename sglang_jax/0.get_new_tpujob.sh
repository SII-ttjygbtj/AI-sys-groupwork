#!/bin/bash -x


sky launch tpu.yaml -y --no-use-spot --infra=gcp -i 9999 --disk-size 1024
