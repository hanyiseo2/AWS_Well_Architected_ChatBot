import os
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = "korea-sw-16-chatbot-s3"
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)

def upload_file(local_path: str, s3_key: str):
    """로컬 파일 → S3 업로드"""
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"✅ 업로드 완료: {local_path} → s3://{BUCKET_NAME}/{s3_key}")
        return True
    except ClientError as e:
        print(f"❌ 업로드 실패: {e}")
        return False


def download_file(s3_key: str, local_path: str):
    """S3 → 로컬 다운로드"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"✅ 다운로드 완료: s3://{BUCKET_NAME}/{s3_key} → {local_path}")
        return True
    except ClientError as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

'''
def upload_directory(local_dir: str, s3_prefix: str):
    """폴더 전체 업로드"""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            upload_file(local_path, s3_key)


def download_directory(s3_prefix: str, local_dir: str):
    """S3 폴더 전체 다운로드"""
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix)
        
        if "Contents" not in response:
            print(f"⚠️ S3에 {s3_prefix} 경로가 비어있습니다.")
            return False
        
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_path = os.path.join(local_dir, relative_path)
            download_file(s3_key, local_path)
        
        return True
    except ClientError as e:
        print(f"❌ 다운로드 실패: {e}")
        return False'''


# 테스트
if __name__ == "__main__":
    # PDF 업로드 테스트
    upload_file(
        "./docs/wellarchitected-framework.pdf",
        "documents/wellarchitected-framework.pdf"
    )
