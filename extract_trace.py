
import tarfile, os
src = r"D:\Backup-May2025\Desktop\PHD\DC MEETING\MLBMFS\code\mlbmfs_philly_pipeline\trace-data.tar.gz"
dest = r"D:\Backup-May2025\Desktop\PHD\DC MEETING\MLBMFS\code\mlbmfs_philly_pipeline\philly_trace_extracted"
os.makedirs(dest, exist_ok=True)
with tarfile.open(src, "r:gz") as tar:
    tar.extractall(dest)
print("✅ Extraction complete:", dest)
