export PROJECT_ID=early-exit-transformer-network
export TPU_NAME=node-2
export ZONE=europe-west4-b
export ACCELERATOR_TYPE=v5litepod-16
export RUNTIME_VERSION=v2-alpha-tpuv5-lite

gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --accelerator-type=${ACCELERATOR_TYPE} \
    --version=${RUNTIME_VERSION}

#install compatible versions of torch and torch-xla 
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html"



#clone xla onto all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="git clone https://github.com/pytorch/xla.git"

#install transformers on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="python3 -m pip install --upgrade pip && python3 -m pip install --user transformers"


#upload script to all workers
gcloud compute tpus tpu-vm scp /home/mikexi/projects/signll_code/src/gemma_test/gemma_3_tpu_test.py ${TPU_NAME}:~ \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all

#run script
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="PJRT_DEVICE=TPU python3 ~/gemma_3_tpu_test.py"
