# OpenJet on a rented GPU

An image with OpenJet, its engine and Qwen3.8-27B Q4_K_M already inside, published as a public template on Runpod and Vast.
Each user rents the GPU on their own provider account and pays the provider directly. OpenJet never rents, bills for or
resells compute, which is how this stays inside the providers' terms (see [Terms](#terms)).

**Status: not yet run on a GPU or on either provider.** The token proxy has tests (`tests/test_cloud_auth_proxy.py`);
the image has not been built. Do the checks under [Publish](#publish) before linking to it.

## What happens on a machine

1. The provider pulls the image. Nothing is downloaded from Hugging Face or GitHub at startup, but the image is about
   18 GB, so a host that doesn't have it cached takes as long as its connection needs (not measured).
2. The entrypoint joins the model file and starts the engine on loopback, with a token proxy on port 8080.
3. The token is printed in the container log and saved in `/root/.openjet-api-token`. Set `OPENJET_API_TOKEN` to choose it.

The engine server doesn't check credentials, so never expose its port (18081) directly.

## Connect

On Runpod, the endpoint is `https://<pod id>-8080.proxy.runpod.net`. Runpod's proxy closes a connection after 100 seconds
with no response, and reading a very long prompt takes longer than that (82 s for 98K tokens on an RTX 3090, before any
queueing), so for long contexts use an SSH tunnel to the pod instead.

On Vast, use an SSH tunnel: the mapped port is plain HTTP, so a direct connection would send your token and code unencrypted.

```bash
ssh -p <port> -L 8080:localhost:8080 root@<host>     # then use http://localhost:8080
```

Claude Code:

```bash
export ANTHROPIC_BASE_URL=<endpoint>
export ANTHROPIC_AUTH_TOKEN=<token>
export ANTHROPIC_MODEL=Qwen3.8-27B-Q4_K_M
export ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-Q4_K_M
claude
```

Codex, OpenCode and OpenClaw take the settings in [docs/clients.md](../../docs/clients.md), with the endpoint above and the
token as the API key. To run OpenJet's own agent on the machine, SSH in and run `openjet-agent`. The endpoint stops while it runs.

Stop or delete the machine in the provider's console when you're done: the provider bills for it until then.

## Publish

```bash
docker login ghcr.io
deploy/cloud/build.sh models/Qwen3.8-27B-Q4_K_M.gguf ghcr.io/l-forster/openjet-cloud:0.5.4-qwen3.8-27b-q4km          # local image
docker run --rm --gpus all -p 8080:8080 ghcr.io/l-forster/openjet-cloud:0.5.4-qwen3.8-27b-q4km                       # test on the 3090
deploy/cloud/build.sh models/Qwen3.8-27B-Q4_K_M.gguf ghcr.io/l-forster/openjet-cloud:0.5.4-qwen3.8-27b-q4km --push   # publish
```

Make the GHCR package public, then create the templates.

- **Runpod:** create a public template from [runpod-template.json](runpod-template.json) in the console, or with
  `POST https://api.runpod.io/v2/templates`. Deploy it on an RTX 3090 and copy the template's deploy link.
- **Vast:** create a template with image path as above, disk 60 GB, port 8080, SSH launch mode, and this on-start command:
  `nohup /opt/openjet/entrypoint.sh > /var/log/openjet.log 2>&1 &`. Filter its search to one RTX 3090, set it public, and
  use **Copy Referral Link** from the template's menu.

Put the links in `NEXT_PUBLIC_RUNPOD_TEMPLATE_URL` and `NEXT_PUBLIC_VAST_TEMPLATE_URL` on the website; `/cloud` turns on
its deploy buttons when they're set.

## Terms

Not legal advice. Read on 2026-11-04:

- **Runpod** bars reselling credits and commercial exploitation of the service without its approval, but invites public
  templates: template creators earn 1 % of spend in credits. Its partners page lists a reseller track for anything further.
- **Vast** bars running a service bureau or reselling without a separate written agreement, but invites public templates
  with a 3 % referral. Since 15 September 2026 it bars paid ads that link to vast.ai or use its trademarks.
- **Reselling GPU hours yourself** (one account, your invoice to the customer) needs those written agreements first.
- **Model:** Apache 2.0 allows redistribution; the image carries the license and [NOTICE](NOTICE). **OpenJet** is AGPL-3.0,
  so the image's source stays public.
- **Privacy:** Vast hosts are independent contractors, and Vast disclaims responsibility for them. Don't describe either
  provider's community machines as private.
