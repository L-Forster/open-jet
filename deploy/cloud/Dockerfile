# OpenJet on a rented RTX 3090: the engine and Qwen3.8-27B Q4_K_M are in the image. Build it with build.sh.
FROM python:3.12-slim-bookworm AS runtime

ARG OPENJET_VERSION=0.5.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git libgomp1 openssh-server tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "open-jet==${OPENJET_VERSION}"

# The sm_86 engine (RTX 3090, 3090 Ti, A5000, A6000), tagged the way `openjet setup` tags a prebuilt install.
RUN set -eu; \
    root="$(python -c 'from src.app_paths import openjet_install_root; print(openjet_install_root())')"; \
    curl -fsSL "https://github.com/L-Forster/open-jet/releases/download/v${OPENJET_VERSION}/openjet-engine-linux-x64-sm86.tar.gz" \
        | tar -xz -C "$root/megakernel"; \
    tag="{\"arch\": \"sm_86\", \"prebuilt\": \"${OPENJET_VERSION}\"}"; \
    echo "$tag" > "$root/megakernel/.openjet-build.json"; \
    echo "$tag" > "$root/megakernel/.openjet-build-mixed.json"; \
    test -f "$root/megakernel/libqwen38mk.so"; \
    test -f "$root/megakernel/draft_vocab_32k.bin"; \
    test -f "$root/megakernel/server/mk_server.py"; \
    ln -s "$root" /opt/openjet-install

COPY auth_proxy.py entrypoint.sh openjet-agent llama-server-unavailable NOTICE /opt/openjet/
COPY config.yaml /opt/openjet-install/config.yaml
RUN chmod +x /opt/openjet/entrypoint.sh /opt/openjet/openjet-agent /opt/openjet/llama-server-unavailable \
    && ln -s /opt/openjet/openjet-agent /usr/local/bin/openjet-agent \
    && cp /usr/share/common-licenses/Apache-2.0 /opt/openjet/LICENSE-Qwen3.8-27B-Apache-2.0

ENV OPENJET_TELEMETRY_NO_PROMPT=1
EXPOSE 8080 22
ENTRYPOINT ["/opt/openjet/entrypoint.sh"]

FROM runtime AS full
# One layer per part: registries can limit layer size, and hosts pull layers in parallel.
COPY model.part-00 /opt/openjet/model/parts/
COPY model.part-01 /opt/openjet/model/parts/
COPY model.part-02 /opt/openjet/model/parts/
COPY model.part-03 /opt/openjet/model/parts/
