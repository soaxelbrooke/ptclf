#!/usr/bin/env bash

export DISK_SIZE=30GB
export GPU_TYPE=nvidia-tesla-k80
export MACHINE_TYPE=n1-highmem-4
export ZONE=us-west1-b
export INST_NAME=ptclf-$(uuidgen | cut -c 1-4)
echo "Creating preemptible GCE GPU $MACHINE_TYPE instance with 1 $GPU_TYPE named '$INST_NAME' in $ZONE..."

gcloud compute instances create $INST_NAME \
    --preemptible \
    --zone $ZONE \
    --machine-type $MACHINE_TYPE \
    --boot-disk-size $DISK_SIZE \
    --accelerator type=$GPU_TYPE,count=1 \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE \
    --scopes storage-rw,https://www.googleapis.com/auth/source.read_only \
    --metadata-from-file startup-script='gce-startup.sh'

