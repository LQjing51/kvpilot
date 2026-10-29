#!/usr/bin/env bash
set -Eeuo pipefail

HEAD_IP="$(hostname -I | awk '{print $1}')"

if [[ -z "${HEAD_IP}" ]]; then
    echo "Cannot determine head IP" >&2
    exit 1
fi

export DYNAMO_HEAD_IP="${HEAD_IP}"
export ETCD_ENDPOINTS="http://${HEAD_IP}:2379"
export NATS_SERVER="nats://${HEAD_IP}:4222"

mkdir -p \
    /var/lib/dynamo/etcd \
    /var/lib/dynamo/nats \
    /var/log/dynamo

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

cleanup() {
    trap - EXIT INT TERM

    kill -TERM \
        "${ETCD_PID:-}" \
        "${NATS_PID:-}" \
        2>/dev/null || true

    wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 0' INT TERM

etcd \
    --name dynamo-etcd \
    --data-dir /var/lib/dynamo/etcd \
    --listen-client-urls "http://0.0.0.0:2379" \
    --advertise-client-urls "http://${HEAD_IP}:2379" \
    --listen-peer-urls "http://127.0.0.1:2380" \
    --initial-advertise-peer-urls "http://127.0.0.1:2380" \
    --initial-cluster "dynamo-etcd=http://127.0.0.1:2380" \
    --initial-cluster-state new \
    >/var/log/dynamo/etcd.log 2>&1 &
ETCD_PID=$!

nats-server \
    --addr 0.0.0.0 \
    --port 4222 \
    --http_port 8222 \
    --jetstream \
    --store_dir /var/lib/dynamo/nats \
    >/var/log/dynamo/nats.log 2>&1 &
NATS_PID=$!

ready=false

for _ in $(seq 1 60); do
    etcd_ready=false
    nats_ready=false

    if ETCDCTL_API=3 etcdctl \
        --endpoints "http://127.0.0.1:2379" \
        endpoint health >/dev/null 2>&1; then
        etcd_ready=true
    fi

    if (
        exec 3<>"/dev/tcp/127.0.0.1/4222"
    ) 2>/dev/null; then
        nats_ready=true
    fi

    if [[ "${etcd_ready}" == true && "${nats_ready}" == true ]]; then
        ready=true
        break
    fi

    if ! kill -0 "${ETCD_PID}" 2>/dev/null; then
        echo "etcd exited unexpectedly:" >&2
        tail -n 100 /var/log/dynamo/etcd.log >&2 || true
        exit 1
    fi

    if ! kill -0 "${NATS_PID}" 2>/dev/null; then
        echo "NATS exited unexpectedly:" >&2
        tail -n 100 /var/log/dynamo/nats.log >&2 || true
        exit 1
    fi

    sleep 1
done

if [[ "${ready}" != true ]]; then
    echo "Timed out waiting for etcd/NATS" >&2
    exit 1
fi

echo
echo "=============================================="
echo "Dynamo head environment is ready"
echo "HEAD_IP=${HEAD_IP}"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
echo "NATS_SERVER=${NATS_SERVER}"
echo
echo "Use this IP when starting the worker:"
echo "${HEAD_IP}"
echo "=============================================="
echo

wait -n "${ETCD_PID}" "${NATS_PID}"