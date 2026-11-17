# Windows build (the Makefile's targets with MSVC 2022 + nvcc): qwen38mk.dll (server engine), qwen38_mk.exe (CLI),
# qwen38vision.dll (CPU image encoder), tools\*.exe. The engine is compiled with the same nvcc flags as the Makefile.
#   .\build.ps1                         everything, the served configuration
#   .\build.ps1 -Targets vision         only the vision library + its tools (no CUDA compile)
# -Llama: llama.cpp built with CMake (BUILD_SHARED_LIBS=ON; default ..\llama.cpp). -Cuda: the CUDA toolkit (default
# $env:CUDA_PATH). -Arch: e.g. sm_86 (default: the first GPU's compute capability).
param(
    [string] $Llama = (Join-Path $PSScriptRoot "..\llama.cpp"),
    [string] $Cuda = $env:CUDA_PATH,
    [string] $Arch = "",
    [string] $MkFlags = "",
    [ValidateSet("all", "engine", "vision", "tools")] [string[]] $Targets = @("all")
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Llama = (Resolve-Path $Llama).Path
if (-not $Cuda -or -not (Test-Path (Join-Path $Cuda "bin\nvcc.exe"))) { throw "CUDA toolkit not found (pass -Cuda or set CUDA_PATH)" }
$nvcc = Join-Path $Cuda "bin\nvcc.exe"
$want = { param($t) $Targets -contains "all" -or $Targets -contains $t }

# MSVC 2022 environment (CUDA 13.2 does not support VS 2026's compiler yet)
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vs) { throw "Visual Studio 2022 (Build Tools) with the C++ x64 tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >NUL && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }

if (-not $Arch) {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    $Arch = if ($cc) { "sm_" + ($cc -replace "[^0-9]", "") } else { "sm_86" }
}
$Arch = "sm_" + ($Arch -replace "^sm_", "")

$llInc = @("-I$Llama\include", "-I$Llama\ggml\include")
$llLib = @("$Llama\build\src\llama.lib", "$Llama\build\ggml\src\ggml-base.lib")
foreach ($f in $llLib) { if (-not (Test-Path $f)) { throw "missing ${f}: build llama.cpp first (README, Windows)" } }
$hostFlags = "/EHsc,/MD,/O2,/std:c++17,/D_CRT_SECURE_NO_WARNINGS,/D_CRT_NONSTDC_NO_WARNINGS,/wd4996"
$cl = @("/nologo", "/O2", "/EHsc", "/MD", "/std:c++17", "/D_CRT_SECURE_NO_WARNINGS", "/D_CRT_NONSTDC_NO_WARNINGS", "/wd4068")

function Run([string] $exe, [string[]] $a, [string] $errFile = "") {
    Write-Host "> $exe $($a -join ' ')" -ForegroundColor DarkGray
    # (the exit code decides: under Stop, a warning on stderr is a terminating error once the output is redirected)
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # with errFile: stdout and stderr both to the file (nvcc reports compile errors on stdout, ptxas -v on stderr)
        if ($errFile) { & $exe @a *> $errFile } else { & $exe @a }
    } finally {
        $ErrorActionPreference = $eap
    }
    if ($LASTEXITCODE -ne 0) {
        if ($errFile) { Get-Content $errFile | Where-Object { $_ -match "error" } | Select-Object -First 20 }
        throw "$exe failed ($LASTEXITCODE)$(if ($errFile) { ", see $errFile" })"
    }
}

# exports of a C API: every extern "C" function defined at column 0 whose name starts with the prefix
function Write-Def([string] $src, [string] $prefix, [string] $dll, [string] $out) {
    $names = Select-String -Path $src -Pattern "^[A-Za-z_][\w\s\*]*?\b($prefix\w*)\s*\(" -AllMatches |
        ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique
    @("LIBRARY $dll", "EXPORTS") + ($names | ForEach-Object { "    $_" }) | Set-Content -Encoding ascii $out
}

function Copy-LlamaDlls([string] $dir) {
    Get-ChildItem "$Llama\build\bin\*.dll" | Where-Object { $_.Name -match "^(llama|ggml.*|mtmd)\.dll$" } |
        ForEach-Object { Copy-Item $_.FullName $dir -Force }
}

New-Item -ItemType Directory -Force build_win | Out-Null
if (& $want "engine") {
    Run "cl.exe" ($cl + @("/c", "compat_win32.cpp", "/Fobuild_win\compat_win32.obj"))
    $nvBase = @("-O3", "-std=c++17", "-arch=$Arch", "-lineinfo", "-Xptxas", "-dlcm=cg", "-Xptxas", "-v", "-Xcompiler", $hostFlags) +
              ($MkFlags -split "\s+" | Where-Object { $_ }) + $llInc
    # an object is rebuilt unless it is newer than its sources (the Makefile's lists) and was built with these flags: a
    # change on the host side does not redo the device TU's 20-minute compile
    function Build-Cu([string] $src, [string[]] $deps, [string] $log) {
        $name = [IO.Path]::GetFileNameWithoutExtension($src)
        $obj, $stamp = "build_win\$name.obj", "build_win\$name.flags"
        $nv = $nvBase + @("-c", $src, "-o", $obj)
        $newest = (Get-Item (@($src) + $deps) | Measure-Object -Property LastWriteTime -Maximum).Maximum
        $fresh = (Test-Path $obj) -and (Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq ($nv -join " ")) -and
                 ((Get-Item $obj).LastWriteTime -gt $newest)
        if ($fresh) {
            Write-Host "$name.obj is up to date"
            return
        }
        Remove-Item $stamp -ErrorAction SilentlyContinue
        $t0 = Get-Date
        Run $nvcc $nv $log
        Set-Content -Encoding ascii $stamp ($nv -join " ")
        Write-Host ("$src compiled in {0:n0} s" -f ((Get-Date) - $t0).TotalSeconds)
    }
    # the device code and its launchers; ptxas register / spill report for every phase function goes to ptxas.log (as with make)
    Build-Cu "qwen38_mk.cu" @("mk_shared.cuh", "mk_launch.h", "qwen38_mk.h", "gguf.h", "compat_win32.h", "prefill.cuh",
                              "prefill_gemm.cuh", "prefill_attn.cuh", "wv.cuh", "wv_dev.cuh") "ptxas.log"
    # the host side, same flags (the shared definitions must agree)
    $hostSrc = @("engine.cu", "ck_pool.cu", "loader.cu", "prefill_host.cu", "state_files.cu", "host_store.cu", "lanes.cu")
    $hostHdr = @("engine_impl.h", "mk_shared.cuh", "mk_launch.h", "qwen38_mk.h", "gguf.h", "compat_win32.h", "loader.cuh", "wv_host.cuh")
    foreach ($s in $hostSrc) { Build-Cu $s $hostHdr "" }
    $engineObj = @("build_win\qwen38_mk.obj") + ($hostSrc | ForEach-Object { "build_win\" + [IO.Path]::GetFileNameWithoutExtension($_) + ".obj" })
    Write-Def "capi.cpp" "mk_" "qwen38mk.dll" "build_win\qwen38mk.def"
    # the build's commit and engine flags for capi.cpp's mk_build_info (the server's /props)
    $commit = "unknown"
    try {
        $c = (& git rev-parse --short=12 HEAD 2>$null)
        if ($LASTEXITCODE -eq 0 -and $c) {
            $commit = $c.Trim()
            & git diff --quiet HEAD -- 2>$null
            if ($LASTEXITCODE -ne 0) { $commit += "-dirty" }
        }
    } catch { }
    Set-Content -Encoding ascii "build_win\mk_build_info.h" @("#define MK_BUILD_COMMIT `"$commit`"",
        "#define MK_BUILD_FLAGS `"$(($MkFlags -split '\s+' | Where-Object { $_ }) -join ' ')`"")
    $link = @("-O3", "-std=c++17", "-arch=$Arch", "-Xcompiler", $hostFlags) + $llInc
    # /EHc-: libllama's extern "C" API throws C++ exceptions, and under nvcc's -EHsc MSVC drops capi.cpp's catches around
    # its calls (/EHs does not undo the c; cl warns D9025)
    Run $nvcc ($link + @("-Xcompiler", "/EHc-", "-shared", "-Ibuild_win", "capi.cpp") + $engineObj + @("build_win\compat_win32.obj") +
               $llLib + @("-Xlinker", "/DEF:build_win\qwen38mk.def", "-o", "qwen38mk.dll"))
    Run $nvcc ($link + @("main.cu") + $engineObj + @("build_win\compat_win32.obj") + $llLib + @("-o", "qwen38_mk.exe"))
    Copy-LlamaDlls $PSScriptRoot
}
if (& $want "tools") {
    foreach ($t in @("vocab_freq", "llama_ref")) {
        Run "cl.exe" ($cl + @("/I$Llama\include", "/I$Llama\ggml\include", "tools\$t.cpp", "/Fobuild_win\$t.obj",
                              "/Fetools\$t.exe", "/link") + $llLib)
    }
    Copy-LlamaDlls (Join-Path $PSScriptRoot "tools")
}
if (& $want "vision") {
    $vinc = @("/I$Llama\include", "/I$Llama\ggml\include", "/I$Llama\tools\mtmd", "/I$Llama\vendor")
    $vlib = @("$Llama\build\tools\mtmd\mtmd.lib", "$Llama\build\src\llama.lib", "$Llama\build\ggml\src\ggml.lib",
              "$Llama\build\ggml\src\ggml-base.lib")
    # vit_cpu: AVX2 / FMA in its own object (the DLL still loads on older CPUs, which then use libmtmd); MSVC's default
    # /fp:precise does not contract a * b + c into FMAs, which keeps the preprocessing's arithmetic llama.cpp's
    Run "cl.exe" ($cl + $vinc + @("/arch:AVX2", "/fp:precise", "/c", "vision\vit_cpu.cpp", "/Fobuild_win\vit_cpu.obj"))
    Run "cl.exe" ($cl + @("/c", "vision\win32.cpp", "/Fobuild_win\vision_win32.obj"))
    Write-Def "vision\mk_vision.cpp" "mkv_" "qwen38vision.dll" "build_win\qwen38vision.def"
    Run "cl.exe" ($cl + $vinc + @("/LD", "vision\mk_vision.cpp", "build_win\vit_cpu.obj", "build_win\vision_win32.obj",
                                  "/Fobuild_win\mk_vision.obj", "/Feqwen38vision.dll", "/link", "/DEF:build_win\qwen38vision.def") + $vlib)
    Run "cl.exe" ($cl + @("/Ivision", "vision\mkv_cli.cpp", "/Fobuild_win\mkv_cli.obj", "/Fetools\mkv_cli.exe", "/link", "qwen38vision.lib"))
    Run "cl.exe" ($cl + $vinc + @("tools\vision_ref.cpp", "/Fobuild_win\vision_ref.obj", "/Fetools\vision_ref.exe", "/link") + $vlib)
    Copy-LlamaDlls $PSScriptRoot
    Copy-Item qwen38vision.dll tools -Force
    Copy-LlamaDlls (Join-Path $PSScriptRoot "tools")
}
Write-Host "build done ($Arch, MKFLAGS '$MkFlags')" -ForegroundColor Green
