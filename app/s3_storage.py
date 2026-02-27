import boto3
import os

S3_BUCKET = "finance-ai-vector-storage"
FAISS_DIR = "faiss_index"

s3 = boto3.client("s3")


def upload_faiss_to_s3():
    for root, _, files in os.walk(FAISS_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, FAISS_DIR)
            s3.upload_file(local_path, S3_BUCKET, s3_path)


def download_faiss_from_s3():
    os.makedirs(FAISS_DIR, exist_ok=True)

    response = s3.list_objects_v2(Bucket=S3_BUCKET)

    if "Contents" not in response:
        return

    for obj in response["Contents"]:
        s3.download_file(S3_BUCKET, obj["Key"], os.path.join(FAISS_DIR, obj["Key"]))
