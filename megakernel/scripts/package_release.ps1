# Package a Windows release (zip + sha256) from a finished build.ps1 of this tree (all targets).
#   .\scripts\package_release.ps1 -Version v2026.10.22 [-Llama ..\llama.cpp] [-Out dist]
# Bundles the Visual C++ runtime from the Visual Studio 2022 redistributable folder, so the package runs on a PC without
# Visual Studio or the VC++ redistributable installed.
param(
    [Parameter(Mandatory = $true)] [string] $Version,
    [string] $Llama = (Join-Path $PSScriptRoot "..\..\llama.cpp"),
    [string] $Out = (Join-Path $PSScriptRoot "..\dist")
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Llama = (Resolve-Path $Llama).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
Set-Location $Root

$binRoot = @("qwen38mk.dll", "qwen38_mk.exe", "qwen38vision.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "mtmd.dll")
$binTools = @("llama_ref.exe", "vocab_freq.exe", "mkv_cli.exe", "vision_ref.exe")
foreach ($f in $binRoot) { if (-not (Test-Path $f)) { throw "missing ${f}: run build.ps1 first" } }
foreach ($f in $binTools) { if (-not (Test-Path "tools\$f")) { throw "missing tools\${f}: run build.ps1 first" } }
$info = Get-Content "build_win\mk_build_info.h"
$commit = ($info | Select-String '^#define MK_BUILD_COMMIT "(.*)"').Matches[0].Groups[1].Value
$flags = ($info | Select-String '^#define MK_BUILD_FLAGS "(.*)"').Matches[0].Groups[1].Value
$head = (git rev-parse --short=12 HEAD).Trim()
if ($commit -ne $head) { throw "the build is of '$commit', HEAD is ${head}: run build.ps1 again" }

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -version "[17.0,18.0)" -products * -property installationPath | Select-Object -First 1
$redist = Get-ChildItem (Join-Path $vs "VC\Redist\MSVC") -Directory | Where-Object { Test-Path "$($_.FullName)\x64\Microsoft.VC143.CRT" } |
    Sort-Object { [version]$_.Name } -ErrorAction SilentlyContinue | Select-Object -Last 1
if (-not $redist) { throw "Visual Studio 2022 redistributable folder (VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT) not found" }
$vcrt = @("$($redist.FullName)\x64\Microsoft.VC143.CRT\msvcp140.dll", "$($redist.FullName)\x64\Microsoft.VC143.CRT\vcruntime140.dll",
          "$($redist.FullName)\x64\Microsoft.VC143.CRT\vcruntime140_1.dll", "$($redist.FullName)\x64\Microsoft.VC143.OpenMP\vcomp140.dll")
foreach ($f in $vcrt) { if (-not (Test-Path $f)) { throw "missing $f" } }

# the GPU code the build holds (build.ps1 -Arch: the first GPU's, unless given)
$cuobjdump = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin\cuobjdump.exe" } else { "cuobjdump.exe" }
$archs = (& $cuobjdump --list-elf "qwen38mk.dll" | Select-String -AllMatches 'sm_\d+[a-z]?' | ForEach-Object { $_.Matches.Value } |
          Sort-Object -Unique) -join "-"
if (-not $archs) { throw "no GPU code found in qwen38mk.dll (cuobjdump --list-elf)" }
$gpuNote = if ($archs -eq "sm_86") { " (compute capability 8.6; 8.9 runs it, untested)" } else { "" }

$name = "qwen38-megakernel-$Version-windows-x64-$($archs -replace '_', '')"
$stage = Join-Path $Out $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
foreach ($d in @("tools", "licenses", "server", "scripts")) { New-Item -ItemType Directory -Force (Join-Path $stage $d) | Out-Null }
Copy-Item $binRoot $stage
Copy-Item ($binTools | ForEach-Object { "tools\$_" }) "$stage\tools"
# the tools load the same DLLs from their own folder
Copy-Item ($binRoot | Where-Object { $_ -match "\.dll$" -and $_ -ne "qwen38mk.dll" }) "$stage\tools"
Copy-Item $vcrt $stage
Copy-Item $vcrt "$stage\tools"
Copy-Item "server\*.py" "$stage\server"
Copy-Item "scripts\serve.bat", "scripts\exactness.py", "scripts\l2_sweep.py" "$stage\scripts"
Copy-Item -Recurse "docs" $stage
Copy-Item @("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt", "draft_vocab_32k.bin") $stage
Copy-Item "$Llama\LICENSE" "$stage\licenses\llama.cpp-LICENSE"
Get-ChildItem "$Llama\licenses" -File -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName "$stage\licenses\llama.cpp-$($_.Name)" }
function Write-Tail([string] $src, [string] $start, [string] $dst) {
    $lines = Get-Content $src
    $i = -1
    for ($k = 0; $k -lt $lines.Count; $k++) { if ($lines[$k].StartsWith($start)) { $i = $k; break } }
    if ($i -lt 0) { throw "license text not found in $src" }
    $lines[$i..($lines.Count - 1)] | Set-Content -Encoding utf8 $dst
}
Write-Tail "$Llama\vendor\stb\stb_image.h" "This software is available under 2 licenses" "$stage\licenses\stb_image-LICENSE"
Write-Tail "$Llama\vendor\miniaudio\miniaudio.h" "This software is available as a choice of the following licenses" "$stage\licenses\miniaudio-LICENSE"

$nvccVer = if ($env:CUDA_PATH) { (& "$env:CUDA_PATH\bin\nvcc.exe" --version | Select-Object -Last 1) } else { "CUDA 13.2" }
$llamaTag = (git -C $Llama describe --tags 2>$null)
@(
    "qwen38-megakernel $Version, Windows x64",
    "commit     $commit (https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/tree/$commit)",
    "MKFLAGS    $flags",
    "GPU code   $($archs -replace '-', ', ')$gpuNote",
    "CUDA       $nvccVer",
    "compiler   MSVC $($redist.Name) (Visual Studio 2022)",
    "llama.cpp  $llamaTag (https://github.com/ggml-org/llama.cpp/tree/$llamaTag), CPU only",
    "needs      Windows 10 / 11 x64, an NVIDIA driver with CUDA 13.2 support, an x86-64 CPU with AVX2",
    "built      $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mmZ'))"
) | Set-Content -Encoding utf8 "$stage\BUILD_INFO.txt"
@(
    "qwen38-megakernel: Qwen3.8-27B on one 24 GB NVIDIA GPU, with an OpenAI / Anthropic / OpenAI Responses compatible server.",
    "",
    "1. Python 3.10+:   pip install -r requirements.txt",
    "2. The model:      pip install huggingface_hub",
    "                   hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir %USERPROFILE%\models",
    "3. Run (cmd):      scripts\serve.bat                    (http://127.0.0.1:8080, 120k context)",
    "                   scripts\serve.bat 8080 64000         (port, context)",
    "                   set MK_KV=K*=6,V*=5",
    "                   set MK_KV_WIN=2048",
    "                   scripts\serve.bat 8080 262144        (the full 262k context, preset D)",
    "                   set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf   (another model path)",
    "",
    "Client settings: docs\CLIENTS.md. Every setting: docs\CONFIGURATION.md. What is in this package and what it needs:",
    "BUILD_INFO.txt, THIRD_PARTY_NOTICES.md."
) | Set-Content -Encoding utf8 "$stage\QUICKSTART.txt"

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Set-Content -Encoding ascii "$zip.sha256"
Write-Host ("{0} ({1:n0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)) -ForegroundColor Green
