export PROJECT_ID=early-exit-transformer-network
export TPU_NAME=node-4
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

#install huggingface datasets library on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install --upgrade datasets"

#upload script to all workers
gcloud compute tpus tpu-vm scp /home/mikexi/projects/signll_code/src/final_tests/training_data_download_test.py ${TPU_NAME}:~ \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all

#https://cloud.google.com/tpu/docs/pytorch-pods - gcp documentation says use 2.5.0, but use 2.6.0 because there was a vulnerability in torch.load
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=all \
    --command="pip install torch~=2.6.0 torch_xla[tpu]~=2.6.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html"
#run script
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="PJRT_DEVICE=TPU python3 ~/training_data_download_test.py"

  #kill all user run processes: pkill -f -u mikexi python3

#install fasttext on all tpu workers (DO NOT USE ANYMORE)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install fasttext"

#install requests on all tpu workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install requests"

#install protobuf compiler on all TPUs (if you really need it, often it is default loaded)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="sudo apt-get update && sudo apt-get install -y protobuf-compiler"

#install gcld3
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install gcld3"


