@echo off
REM Serve Qwen3.8-27B through the megakernel's OpenAI-compatible server (native Windows). Usage: scripts\serve.bat [port] [ctx]
REM Defaults as scripts/serve.sh: port 8080, 120,000 tokens of context, q8_0 KV cache, 7 MTP drafts, exact reuse on, images
REM on when the mmproj file and qwen38vision.dll exist. Settings (environment):
REM   MK_MODEL            the GGUF (default %USERPROFILE%\models\Qwen3.8-27B-UD-Q4_K_XL.gguf)
REM   MK_MMPROJ           the vision projector (default: mmproj-BF16.gguf next to the model; MK_MMPROJ=off turns images off)
REM   MK_IMAGE_MAX_TOKENS tokens per image (default: the mmproj's, 4096)
REM   MK_VIDEO_FPS        video input (input_video parts, with the mmproj; needs ffmpeg: on PATH, MK_FFMPEG, or
REM                       pip install imageio-ffmpeg): frames sampled per second (default 4, llama.cpp's)
REM   MK_VIDEO_MAX_TOKENS tokens per video frame pair (default: the image limit); fewer fit longer clips
REM   MK_FFMPEG           the ffmpeg executable (or its directory) for video input
REM   MK_HOST             listen address (default 127.0.0.1; 0.0.0.0 for other machines on the LAN)
REM   MK_ALLOW_HOSTS      host names clients use besides localhost, IP addresses and MK_HOST (comma-separated; * = any)
REM   MK_CORS_ORIGINS     web origins whose pages may call the server (comma-separated; see scripts/serve.sh)
REM   MK_API_KEY          require this key on every request but /health (clients send it as their API key); default none
REM   MK_DRAFT            MTP drafts per cycle (default 7)
REM   MK_EXACT_REUSE=0    turn exact reuse off (see scripts/serve.sh)
REM   MK_SIDE_MAX         side requests up to this many prompt tokens (default 8192; 0 = off; see scripts/serve.sh)
REM   MK_PARK_MIB         host memory for parked conversations (default 8192 MiB; 0 = off; see scripts/serve.sh)
REM   MK_PF_CAP           prefill chunk tokens (default 2048; 1024 above 96k context; 512 above 240k with your own MK_KV)
REM   MK_KV, MK_KV_WIN    low-bit KV cache, preset D': set MK_KV=K*=6,V*=5,*16=8 and MK_KV_WIN=2048; above 240k
REM                       context, unless set: preset F (MK_KV=K*=5r,V*=5, MK_KV_WIN=2048)
REM   MK_KEEP_WARM_S      seconds the engine keeps the GPU's clocks up after each request (default 120, 0 = off)
REM   MK_DRAFT_Q4=0       the draft head reads lm_head instead of its Q4_K copy (saves 90 MiB; the copy is on
REM                       by default, off above 240k context with your own MK_KV)
REM   MK_MTP              the MTP draft layer from a separate GGUF of the same model (e.g. an mtp-*.gguf)
REM   MK_NO_MTP=1         no MTP layer: plain decoding, depth 0 (also automatic for a model file without one)
REM   MK_CHAT_TEMPLATE_FILE  a Jinja chat template file used instead of the GGUF's (--chat-template-file, as llama-server)
REM   MK_CHAT_TEMPLATE_KWARGS  defaults under every request's chat_template_kwargs: set "MK_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}"
REM   MK_CONTENT_SPECIALS special-token text inside messages: text (default) or parse (as llama-server; see scripts/serve.sh)
REM   MK_REASONING_FORMAT auto (default: the thinking as reasoning_content), none (left in the content), deepseek-legacy
REM   MK_MAX_BODY_MB, MK_MAX_IMAGES, MK_IMAGE_MAX_PIXELS, MK_MAX_VIDEO_FRAMES
REM                       request limits: body size (default 128 MB), images per request (32), pixels per image
REM                       (100,000,000; checked before decoding), video frames per request (256)
REM   MK_MAX_INFLIGHT_MB, MK_MAX_CONNECTIONS
REM                       request bodies in memory at once (default 256 MB), open connections (256): past them a 503
REM   MK_TIMEOUT_S        seconds a connection may make no progress before it closes (default 600, 0 = never)
REM   MK_REQUEST_TIMEOUT_S seconds a request may take to arrive before its connection closes (default 300, 0 = no limit)
REM   MK_PYTHON           Python with jinja2 (default python)
setlocal
set "HERE=%~dp0.."
set "PORT=%~1"
if "%PORT%"=="" set PORT=8080
set "CTX=%~2"
if "%CTX%"=="" set CTX=120000
if not defined MK_MODEL set "MK_MODEL=%USERPROFILE%\models\Qwen3.8-27B-UD-Q4_K_XL.gguf"
if not defined MK_PYTHON set "MK_PYTHON=python"
if not defined MK_HOST set "MK_HOST=127.0.0.1"
if not defined MK_DRAFT set "MK_DRAFT=7"
if not exist "%HERE%\qwen38mk.dll" (
    echo missing %HERE%\qwen38mk.dll: run build.ps1 ^(see README^)
    exit /b 1
)
if not exist "%MK_MODEL%" (
    echo missing model %MK_MODEL% ^(set MK_MODEL^)
    exit /b 1
)
REM above 240k, preset D's KV cache plus Windows' own share of the card (the desktop) passes what the driver keeps
REM resident and the engine pages (decode ~5x slower). Unless MK_KV is set: preset F (0.58 GB less at 262k). With your
REM own MK_KV: a 512-token chunk and the draft head on lm_head (~0.17 GB less, the same output)
set "KVF="
set "KVS="
if %CTX% GTR 240000 if not defined MK_KV (
    set "MK_KV=K*=5r,V*=5"
    if not defined MK_KV_WIN set "MK_KV_WIN=2048"
    set "KVF=1"
    set "KVS= (preset F)"
)
if %CTX% GTR 240000 if not defined KVF if not defined MK_PF_CAP set "MK_PF_CAP=512"
if %CTX% GTR 240000 if not defined KVF if not defined MK_DRAFT_Q4 set "MK_DRAFT_Q4=0"
if %CTX% GTR 96000 if not defined MK_PF_CAP set "MK_PF_CAP=1024"
set "XR=--exact-reuse"
set "XS=on"
if "%MK_EXACT_REUSE%"=="0" (
    set "XR="
    set "XS=off"
)
if not defined MK_MMPROJ for %%M in ("%MK_MODEL%") do set "MK_MMPROJ=%%~dpMmmproj-BF16.gguf"
set "VS=off"
set "VA="
if /i not "%MK_MMPROJ%"=="off" if exist "%MK_MMPROJ%" if exist "%HERE%\qwen38vision.dll" (
    set "VS=on the CPU"
    set VA=--mmproj "%MK_MMPROJ%"
)
if defined VA if defined MK_IMAGE_MAX_TOKENS set VA=%VA% --image-max-tokens %MK_IMAGE_MAX_TOKENS%
if defined VA if defined MK_VIDEO_FPS set VA=%VA% --video-fps %MK_VIDEO_FPS%
if defined VA if defined MK_VIDEO_MAX_TOKENS set VA=%VA% --video-max-tokens %MK_VIDEO_MAX_TOKENS%
if not defined MK_PF_CAP (set "PFC=2048") else (set "PFC=%MK_PF_CAP%")
set "CT="
set "CTS="
if defined MK_CHAT_TEMPLATE_FILE set CT=--chat-template-file "%MK_CHAT_TEMPLATE_FILE%"
if defined MK_CHAT_TEMPLATE_FILE set "CTS=, chat template %MK_CHAT_TEMPLATE_FILE%"
set "DQ="
if "%MK_DRAFT_Q4%"=="0" set "DQ=, draft head on lm_head"
echo megakernel: %MK_MODEL%, ctx %CTX%, KV q8_0 %MK_KV%%KVS%, %MK_DRAFT% MTP drafts%DQ%, prefill chunk %PFC%, exact reuse %XS%, images %VS%%CTS%, http://%MK_HOST%:%PORT%
"%MK_PYTHON%" "%HERE%\server\mk_server.py" -m "%MK_MODEL%" --host %MK_HOST% --port %PORT% -c %CTX% --kv q8 -d %MK_DRAFT% %XR% %VA% %CT%
