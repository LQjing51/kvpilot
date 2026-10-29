#!/usr/bin/env bash
set -Eeuo pipefail

HEAD_IP="${DYNAMO_HEAD_IP:-${1:-}}"

if [[ -z "${HEAD_IP}" ]]; then
    echo "Usage: start-worker.sh <HEAD_IP>" >&2
    echo "Or set DYNAMO_HEAD_IP" >&2
    exit 1
fi

export DYNAMO_HEAD_IP="${HEAD_IP}"
export ETCD_ENDPOINTS="http://${HEAD_IP}:2379"
export NATS_SERVER="nats://${HEAD_IP}:4222"

mkdir -p /var/log/dynamo

# 太极平台起的容器不继承镜像创建时的 ENV，登录 shell 时 PATH 会被 /etc/profile 重建。
{
    printf 'export DYNAMO_HEAD_IP=%q\n' "${DYNAMO_HEAD_IP}"
    printf 'export ETCD_ENDPOINTS=%q\n' "${ETCD_ENDPOINTS}"
    printf 'export NATS_SERVER=%q\n' "${NATS_SERVER}"
    printf 'export RUSTUP_HOME=%q\n' "${RUSTUP_HOME:-/usr/local/rustup}"
    printf 'export CARGO_HOME=%q\n' "${CARGO_HOME:-/usr/local/cargo}"
    printf 'export PATH=%s\n' \
        "${CARGO_HOME:-/usr/local/cargo}/bin:\$PATH"
} >/etc/profile.d/dynamo.sh

chmod 0644 /etc/profile.d/dynamo.sh

echo "Waiting for head at ${HEAD_IP} ..."

ready=false

for _ in $(seq 1 120); do
    etcd_ready=false
    nats_ready=false

    if ETCDCTL_API=3 etcdctl \
        --endpoints "${ETCD_ENDPOINTS}" \
        endpoint health >/dev/null 2>&1; then
        etcd_ready=true
    fi

    if (
        exec 3<>"/dev/tcp/${HEAD_IP}/4222"
    ) 2>/dev/null; then
        nats_ready=true
    fi

    if [[ "${etcd_ready}" == true && "${nats_ready}" == true ]]; then
        ready=true
        break
    fi

    sleep 2
done

if [[ "${ready}" != true ]]; then
    echo "Cannot connect to etcd/NATS on head ${HEAD_IP}" >&2
    echo "Check ports 2379 and 4222 between the two instances." >&2
    exit 1
fi

echo
echo "=============================================="
echo "Dynamo worker environment is ready"
echo "DYNAMO_HEAD_IP=${DYNAMO_HEAD_IP}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "NATS_SERVER=${NATS_SERVER}"
echo "=============================================="
echo

# 暂时不启动 vLLM/Dynamo component，只保持容器运行。
exec sleep infinity