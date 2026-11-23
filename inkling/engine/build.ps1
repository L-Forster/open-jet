# Windows build of OpenJet's Inkling 2 Mini engine (MSVC 2022 + nvcc): inkling.dll.
#   .\build.ps1 -Llama C:\llama.cpp -Arch sm_86
param(
    [string] $Llama = (Join-Path $PSScriptRoot "..\..\llama.cpp"),
    [string] $Cuda = $env:CUDA_PATH,
    [string] $Arch = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Llama = (Resolve-Path $Llama).Path
if (-not $Cuda -or -not (Test-Path (Join-Path $Cuda "bin\nvcc.exe"))) { throw "CUDA toolkit not found (pass -Cuda or set CUDA_PATH)" }
$nvcc = Join-Path $Cuda "bin\nvcc.exe"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vs) { throw "Visual Studio 2022 (Build Tools) with C++ x64 tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >NUL && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }

if (-not $Arch) {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    $Arch = if ($cc) { "sm_" + ($cc -replace "[^0-9]", "") } else { "sm_86" }
}
$Arch = "sm_" + ($Arch -replace "^sm_", "")

$llInc = @("-I$Llama\include", "-I$Llama\ggml\include")
$llLib = @("llama.lib", "ggml-base.lib") | ForEach-Object {
    $hit = Get-ChildItem "$Llama\build" -Recurse -Filter $_ | Select-Object -First 1
    if (-not $hit) { throw "missing $_ under $Llama\build: build llama.cpp first" }
    $hit.FullName
}
$hostFlags = "/EHsc,/MD,/O2,/std:c++17,/D_CRT_SECURE_NO_WARNINGS,/D_CRT_NONSTDC_NO_WARNINGS,/wd4996"
$cl = @("/nologo", "/O2", "/EHsc", "/MD", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS", "/D_CRT_NONSTDC_NO_WARNINGS", "/wd4068")

function Run([string] $exe, [string[]] $a) {
    Write-Host "> $exe $($a -join ' ')" -ForegroundColor DarkGray
    & $exe @a
    if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE)" }
}

function Write-Def([string[]] $srcs, [string[]] $prefixes, [string] $dll, [string] $out) {
    $pat = "^[A-Za-z_][\w\s\*]*?\b((" + ($prefixes -join "|") + ")\w*)\s*\("
    $names = Select-String -Path $srcs -Pattern $pat -AllMatches |
        ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
    @("LIBRARY $dll", "EXPORTS") + ($names | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $out
}

New-Item -ItemType Directory -Force build_win | Out-Null
Run "cl.exe" ($cl + @("/c", "compat_win32.cpp", "/Fobuild_win\compat_win32.obj"))
Run "cl.exe" ($cl + @("/arch:AVX2", "/c", "cpu_experts.cpp", "/Fobuild_win\cpu_experts.obj"))
Run "cl.exe" ($cl + @("/EHc-", "/I$Llama\include", "/I$Llama\ggml\include", "/c", "capi.cpp", "/Fobuild_win\capi.obj"))

$nvBase = @("-O3", "-std=c++17", "-arch=$Arch", "-Xcompiler", $hostFlags)
$cuSrc = @("ik_engine.cu", "ik_kernels.cu", "ik_prefill.cu", "ik_prefill_tc.cu", "ik_dspark.cu", "expert_kernels.cu", "gguf_kernels.cu")
$cuObj = @()
foreach ($s in $cuSrc) {
    $obj = "build_win\" + [IO.Path]::GetFileNameWithoutExtension($s) + ".obj"
    Run $nvcc ($nvBase + @("-c", $s, "-o", $obj))
    $cuObj += $obj
}

Write-Def @("capi.cpp", "ik_engine.cu") @("mk_", "ik_") "inkling.dll" "build_win\inkling.def"
Run $nvcc ($nvBase + @("-shared") + $cuObj + @("build_win\cpu_experts.obj", "build_win\capi.obj", "build_win\compat_win32.obj") +
           $llLib + @("-Xlinker", "/DEF:build_win\inkling.def", "-o", "inkling.dll"))
Get-ChildItem "$Llama\build\bin" -Recurse -Filter "*.dll" | Where-Object { $_.Name -match "^(llama|ggml.*)\.dll$" } |
    ForEach-Object { Copy-Item $_.FullName $PSScriptRoot -Force }
Write-Host "inkling.dll built ($Arch)" -ForegroundColor Green
