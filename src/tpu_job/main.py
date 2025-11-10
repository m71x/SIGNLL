#TODO: first have the model run the following on all workers and display success/failure:
# 1. display all TPU cores available and if possible, their runtime status
# 2. retrieving and displaying current checkpoints for each core
# 3. download model on all workers
# 4. Have each chip download its respective shard of the dataset

#then, actually run the heavy inference part and make sure it can display consistent output as it runs and uploads results to the GCS bucket
#for each input, first make sure it is english. If not, skip