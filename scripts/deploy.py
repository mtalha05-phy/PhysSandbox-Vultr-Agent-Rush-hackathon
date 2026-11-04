"""Deploy PhysSandbox update to the Vultr VM (192.248.159.149)."""

import os
import subprocess
import sys

TARGET_HOST = "192.248.159.149"
TARGET_DIR = "/opt/physsandbox"
TAR_FILE = "physsandbox-v2.tar.gz"


def deploy(password: str = None, key_path: str = None):
    if not os.path.exists(TAR_FILE):
        print(f"Archive {TAR_FILE} not found. Please generate it first.")
        sys.exit(1)

    print(f"Deploying {TAR_FILE} to root@{TARGET_HOST}:{TARGET_DIR}...")

    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no"]

    if key_path:
        ssh_cmd.extend(["-i", key_path])
        scp_cmd.extend(["-i", key_path])

    # Copy tarball
    print("1. Uploading package...")
    scp_full = scp_cmd + [TAR_FILE, f"root@{TARGET_HOST}:/tmp/{TAR_FILE}"]
    res_scp = subprocess.run(scp_full)
    if res_scp.returncode != 0:
        print("Upload failed. Check SSH credentials.")
        return False

    # Extract and rebuild docker containers
    print("2. Extracting and rebuilding containers on remote VM...")
    remote_script = f"""
    mkdir -p {TARGET_DIR}
    tar -xzf /tmp/{TAR_FILE} -C {TARGET_DIR}
    cd {TARGET_DIR}
    docker compose up -d --build
    docker compose ps
    """
    res_ssh = subprocess.run(ssh_cmd + [f"root@{TARGET_HOST}", remote_script])
    if res_ssh.returncode != 0:
        print("Remote rebuild command failed.")
        return False

    print("Successfully deployed update to http://192.248.159.149/!")
    return True


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    deploy(key_path=key)
