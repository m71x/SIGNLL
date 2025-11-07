from google.cloud import storage
client = storage.Client()
buckets = list(client.list_buckets())
print("Buckets you have access to:", [b.name for b in buckets])
