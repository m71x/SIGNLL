# start python virtual environment and install python package manager
sudo apt-get update -y
sudo apt-get install -y python3-venv git
python3 -m venv env
source env/bin/activate
pip install -U pip

#install JAX and hugging face transformers library to download gemma3
pip install jax[tpu]==0.4.33 \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install transformers>=4.45.0 sentencepiece safetensors

#install numpy and pytorch
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'


#install google cloud storage api on each worker:
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install --user google-cloud-storage"

#install sentencepiece on each worker:
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="pip install --user sentencepiece"
