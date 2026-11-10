#!/usr/bin/env bash
# Start the token proxy and OpenJet's engine. Runpod runs this as the image entrypoint; on Vast it is the on-start command.
set -euo pipefail

MODEL=/opt/openjet/model/Qwen3.8-27B-Q4_K_M.gguf
MODEL_BYTES=17106773984
ENGINE=/opt/openjet-install/megakernel
PORT="${OPENJET_PORT:-8080}"
ENGINE_PORT=18081
CONTEXT="${OPENJET_CONTEXT_TOKENS:-100000}"
TOKEN_FILE=/root/.openjet-api-token
PAUSE_FILE=/run/openjet-engine.pause
PID_FILE=/run/openjet-engine.pid

if [ ! -f "$MODEL" ]; then
    echo "Joining the model file (it ships in parts inside the image)..."
    cat /opt/openjet/model/parts/model.part-* > "$MODEL.partial"
    [ "$(stat -c %s "$MODEL.partial")" = "$MODEL_BYTES" ] || { echo "The joined model has the wrong size." >&2; exit 1; }
    mv "$MODEL.partial" "$MODEL"
fi

# An unresolved template placeholder is not a token.
case "${OPENJET_API_TOKEN:-}" in
    ""|*"{{"*) OPENJET_API_TOKEN="$(cat "$TOKEN_FILE" 2>/dev/null || python -c 'import secrets; print(secrets.token_urlsafe(32))')" ;;
esac
export OPENJET_API_TOKEN
(umask 077; printf '%s\n' "$OPENJET_API_TOKEN" > "$TOKEN_FILE")

# Runpod passes the account's key in PUBLIC_KEY. Vast's SSH launch mode runs its own sshd and sets neither variable.
ssh_key="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [ -n "$ssh_key" ]; then
    install -d -m 700 /root/.ssh
    printf '%s\n' "$ssh_key" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null
    install -d -m 755 /run/sshd
    /usr/sbin/sshd -o PasswordAuthentication=no
fi

python /opt/openjet/auth_proxy.py --port "$PORT" --upstream-port "$ENGINE_PORT" &
proxy_pid=$!
trap 'kill "$proxy_pid" 2>/dev/null || true; [ -f "$PID_FILE" ] && kill "$(cat "$PID_FILE")" 2>/dev/null || true' EXIT

cat <<EOF

OpenJet endpoint: port $PORT (/health answers once the model has loaded)
API token:        $OPENJET_API_TOKEN
Saved in:         $TOKEN_FILE

EOF

export LD_LIBRARY_PATH="$ENGINE:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY CUDA_DEVICE_ORDER=PCI_BUS_ID

# `openjet-agent` pauses this loop, because the engine and OpenJet's own agent can't share the GPU's memory.
while kill -0 "$proxy_pid" 2>/dev/null; do
    if [ -e "$PAUSE_FILE" ]; then
        sleep 2
        continue
    fi
    python "$ENGINE/server/mk_server.py" --lib "$ENGINE/libqwen38mk.so" -m "$MODEL" \
        --host 127.0.0.1 --port "$ENGINE_PORT" -c "$CONTEXT" &
    echo $! > "$PID_FILE"
    wait "$!" || echo "The engine exited with status $?." >&2
    rm -f "$PID_FILE"
    [ -e "$PAUSE_FILE" ] || sleep 5
done
