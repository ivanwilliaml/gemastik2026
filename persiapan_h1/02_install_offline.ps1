<#
  02_install_offline.ps1  --  pasang environment dari bundle TANPA internet.
  Jalankan di PC panitia saat Set-Up Work Station (dan sebagai latihan di laptop dengan Wi-Fi mati).

  Cara paling mudah (dari folder tools di bundle / SSD eksternal):
      02_install_offline.bat
  atau:
      powershell -ExecutionPolicy Bypass -File .\02_install_offline.ps1 -Dest C:\gemastik

  Parameter:
    -Bundle        folder bundle (default: satu tingkat di atas folder script ini)
    -Dest          tujuan pemasangan di PC (default C:\gemastik). Venv, model, dan data disalin ke sini.
    -Torch         auto | cu130 | cu126 | cpu   (auto = pilih dari versi CUDA di nvidia-smi)
    -Requirements  file requirements lain (default tools\requirements.in)
    -SkipModels    jangan salin models/nltk_data/data_ext
    -SkipVerify    jangan jalankan 03_verify_offline.py
    -NoTorch       lewati pemilihan dan uji torch (hanya untuk uji script)
    -Force         lanjut walau bundle tidak lengkap / buat ulang venv
#>
param(
    [string]$Bundle = (Split-Path -Parent $PSScriptRoot),
    [string]$Dest = "C:\gemastik",
    [ValidateSet("auto", "cu130", "cu126", "cpu")][string]$Torch = "auto",
    [string]$Requirements = "",
    [switch]$SkipModels,
    [switch]$SkipVerify,
    [switch]$NoTorch,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$TorchVer = "2.14.1"
function Info($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

# Jalankan program native, kembalikan exit code tanpa membuat PowerShell 5.1 salah menganggap stderr sebagai error
function Run([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $exe @arguments | Out-Host; return [int]$LASTEXITCODE } finally { $ErrorActionPreference = $old }
}
function Probe([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $o = & $exe @arguments 2>$null; if ($LASTEXITCODE -eq 0) { return ($o | Out-String).Trim() } else { return $null } }
    catch { return $null } finally { $ErrorActionPreference = $old }
}

# Pilih varian torch dari angka "CUDA Version: X.Y" milik nvidia-smi (fungsi murni agar mudah diuji)
function Pick-TorchVariant([int]$major, [int]$minor, [bool]$hasNvidia) {
    if (-not $hasNvidia) { return "cpu" }
    if ($major -ge 13) { return "cu130" }
    if ($major -eq 12) { return "cu126" }
    return "cpu"
}

if ($MyInvocation.InvocationName -eq ".") { return }   # di-dot-source untuk pengujian fungsi

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Start-Transcript -Path (Join-Path $Dest "install_log.txt") -Append | Out-Null
$Tools = Join-Path $Bundle "tools"
if (-not $Requirements) { $Requirements = Join-Path $Tools "requirements.in" }
Info "Bundle: $Bundle"
Info "Tujuan: $Dest"

# 1. Kelengkapan bundle (menangkap salinan USB yang terpotong)
$man = Join-Path $Bundle "bundle_manifest.csv"
if (Test-Path $man) {
    $bad = @()
    foreach ($r in (Import-Csv $man)) {
        $p = Join-Path $Bundle ($r.path -replace "/", "\")
        if (-not (Test-Path -LiteralPath $p) -or (Get-Item -LiteralPath $p).Length -ne [int64]$r.size) { $bad += $r.path }
    }
    if ($bad.Count -gt 0) {
        Warn ("{0} file hilang/ukuran beda, contoh: {1}" -f $bad.Count, (($bad | Select-Object -First 5) -join ", "))
        if (-not $Force) { Fail "Bundle tidak lengkap. Salin ulang dari laptop, atau pakai -Force untuk lanjut." }
    } else { Ok "Bundle lengkap ($((Import-Csv $man).Count) file)" }
} else { Warn "bundle_manifest.csv tidak ada; kelengkapan tidak dicek" }

# 2. Inventaris mesin
$os = Get-CimInstance Win32_OperatingSystem
$ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$drive = (Split-Path -Qualifier $Dest)
$freeGB = [math]::Round((Get-PSDrive ($drive.TrimEnd(":"))).Free / 1GB, 1)
Info "OS: $($os.Caption) | RAM: $ramGB GB | ruang bebas di ${drive}: $freeGB GB"
$smi = Probe "nvidia-smi" @()
$cudaMaj = 0; $cudaMin = 0; $hasNv = $false
if ($smi) {
    $hasNv = $true
    $gpuName = Probe "nvidia-smi" @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    Info "GPU: $gpuName"
    if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)") { $cudaMaj = [int]$Matches[1]; $cudaMin = [int]$Matches[2]; Info "Driver mendukung CUDA $cudaMaj.$cudaMin" }
} else { Warn "nvidia-smi tidak ditemukan: GPU NVIDIA/driver tidak terdeteksi" }
if ($freeGB -lt 25) { Warn "Ruang bebas di $drive kurang dari 25 GB; pasang + model butuh sekitar 15-20 GB." }

# 3. Python 3.12
$py = $null
$cand = @()
$p1 = Probe "py" @("-3.12", "-c", "import sys;print(sys.executable)"); if ($p1) { $cand += $p1 }
$p2 = Probe "python" @("-c", "import sys;print(sys.executable if sys.version_info[:2]==(3,12) else '')"); if ($p2) { $cand += $p2 }
$cand += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe")
$cand += "C:\Program Files\Python312\python.exe"
foreach ($c in $cand) { if ($c -and (Test-Path $c)) { $v = Probe $c @("-c", "import sys;print('%d.%d' % sys.version_info[:2])"); if ($v -eq "3.12") { $py = $c; break } } }
if (-not $py) {
    $inst = Join-Path $Bundle "installers\python-3.12.10-amd64.exe"
    if (-not (Test-Path $inst)) { Fail "Python 3.12 tidak ada dan installer tidak ditemukan di $inst" }
    Info "Memasang Python 3.12.10 per-pengguna (tanpa admin)..."
    $rc = Run $inst @("/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0", "Include_test=0", "Include_doc=0")
    $py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (-not (Test-Path $py)) { Fail "Python gagal dipasang (kode $rc). Minta bantuan panitia atau pakai Python yang ada." }
}
Ok "Python 3.12: $py"

# 4. VC++ runtime
$vc = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
if ($vc -and $vc.Installed -eq 1) { Ok "VC++ runtime ada" } else {
    $vcx = Join-Path $Bundle "installers\vc_redist.x64.exe"
    if (Test-Path $vcx) {
        Info "Memasang VC++ runtime (butuh admin)..."
        $rc = Run $vcx @("/install", "/quiet", "/norestart")
        if ($rc -in 0, 1638, 3010) { Ok "VC++ runtime terpasang (kode $rc)" } else { Warn "VC++ gagal (kode $rc). Tanpa ini import torch bisa gagal (WinError 126). Minta panitia memasangnya." }
    } else { Warn "VC++ runtime tidak ada dan installer tidak ditemukan." }
}

# 5. Venv
$venv = Join-Path $Dest "env"
$vpy = Join-Path $venv "Scripts\python.exe"
if ((Test-Path $venv) -and $Force) { Info "Menghapus venv lama (-Force)"; Remove-Item $venv -Recurse -Force }
if (-not (Test-Path $vpy)) { Info "Membuat venv di $venv"; $rc = Run $py @("-m", "venv", $venv); if ($rc -ne 0) { Fail "Gagal membuat venv" } }
$wh = Join-Path $Bundle "wheelhouse"
$rc = Run $vpy @("-m", "pip", "install", "--no-index", "--find-links", $wh, "--upgrade", "pip"); if ($rc -ne 0) { Warn "Upgrade pip dilewati" }

# 6. Pilih varian torch
$variant = "none"
if (-not $NoTorch) {
    $variant = if ($Torch -eq "auto") { Pick-TorchVariant $cudaMaj $cudaMin $hasNv } else { $Torch }
    Info "Varian torch: $variant"
}
function Install-All([string]$variant) {
    $pa = @("-m", "pip", "install", "--no-index", "--find-links", $wh)
    if ($variant -ne "none") {
        $alt = Join-Path $Bundle "wheelhouse_$variant"
        if (Test-Path $alt) { $pa += @("--find-links", $alt) }
        $pa += @("-c", (Join-Path $Tools "constraints_$variant.txt"))
    }
    $pa += @("-r", $Requirements)
    return (Run $vpy $pa)
}
$rc = Install-All $variant
if ($rc -ne 0) { Fail "pip install gagal (kode $rc). Lihat pesan di atas / $Dest\install_log.txt" }
Ok "Paket terpasang"

function Test-Cuda {
    $o = Probe $vpy @("-c", "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())")
    return $o
}
if (-not $NoTorch) {
    $res = Test-Cuda
    Info "torch: $res"
    if ($variant -ne "cpu" -and $res -notmatch "True$") {
        Warn "GPU tidak terbaca dengan $variant; mencoba varian lain."
        $order = @("cu130", "cu126", "cpu") | Where-Object { $_ -ne $variant }
        foreach ($v in $order) {
            if ($v -eq "cu130" -and $cudaMaj -lt 13 -and $hasNv) { continue }
            $a = @("-m", "pip", "install", "--no-index", "--force-reinstall", "--no-deps", "--find-links", $wh)
            $alt = Join-Path $Bundle "wheelhouse_$v"; if (Test-Path $alt) { $a += @("--find-links", $alt) }
            $a += "torch==$TorchVer+$v"
            $r2 = Run $vpy $a
            if ($r2 -eq 0) { $res = Test-Cuda; Info "coba ${v}: $res"; if ($res -match "True$" -or $v -eq "cpu") { $variant = $v; break } }
        }
    }
    if ($res -match "True$") { Ok "GPU siap ($res)" } else { Warn "torch berjalan di CPU ($res). Cek driver NVIDIA dan tanya panitia." }
}
$null = Run $vpy @("-m", "pip", "check")
(Probe $vpy @("-m", "pip", "freeze")) | Set-Content (Join-Path $Dest "requirements.lock") -Encoding ascii

# 7. Salin model dan data ke disk lokal PC (lebih cepat dan tidak bergantung pada USB)
if (-not $SkipModels) {
    foreach ($d in @("models", "nltk_data", "data_ext", "docs", "gguf", "llama")) {
        $src = Join-Path $Bundle $d
        if (Test-Path $src) {
            Info "Menyalin $d ..."
            $rc = Run "robocopy" @($src, (Join-Path $Dest $d), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:2", "/W:2")
            if ($rc -ge 8) { Fail "robocopy $d gagal (kode $rc)" }
        }
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "tools") | Out-Null
Copy-Item -Path (Join-Path $Tools "*") -Destination (Join-Path $Dest "tools") -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $Dest "work") | Out-Null

# 8. env.ps1 / env.bat / start_jupyter.bat  (variabel offline harus aktif sebelum import transformers)
$envLines = @(
    "`$env:COMP_ROOT = '$Dest'",
    "`$env:HF_HOME = '$Dest\hf'; `$env:SENTENCE_TRANSFORMERS_HOME = '$Dest\hf\st'; `$env:NLTK_DATA = '$Dest\nltk_data'",
    "`$env:HF_HUB_OFFLINE = '1'; `$env:TRANSFORMERS_OFFLINE = '1'; `$env:HF_DATASETS_OFFLINE = '1'; `$env:HF_EVALUATE_OFFLINE = '1'",
    "`$env:HF_HUB_DISABLE_TELEMETRY = '1'; `$env:DO_NOT_TRACK = '1'; `$env:HF_HUB_DISABLE_UPDATE_CHECK = '1'; `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'",
    "`$env:PYTORCH_ALLOC_CONF = 'expandable_segments:True'; `$env:TOKENIZERS_PARALLELISM = 'false'",
    "& '$Dest\env\Scripts\Activate.ps1'"
)
$envLines | Set-Content (Join-Path $Dest "env.ps1") -Encoding ascii
$bat = @(
    "@echo off",
    "set COMP_ROOT=$Dest",
    "set HF_HOME=$Dest\hf",
    "set SENTENCE_TRANSFORMERS_HOME=$Dest\hf\st",
    "set NLTK_DATA=$Dest\nltk_data",
    "set HF_HUB_OFFLINE=1",
    "set TRANSFORMERS_OFFLINE=1",
    "set HF_DATASETS_OFFLINE=1",
    "set HF_EVALUATE_OFFLINE=1",
    "set HF_HUB_DISABLE_TELEMETRY=1",
    "set DO_NOT_TRACK=1",
    "set PYTORCH_ALLOC_CONF=expandable_segments:True",
    "set TOKENIZERS_PARALLELISM=false",
    "call `"$Dest\env\Scripts\activate.bat`""
)
$bat | Set-Content (Join-Path $Dest "env.bat") -Encoding ascii
@("@echo off", "call `"$Dest\env.bat`"", "cd /d `"$Dest\work`"", "jupyter lab") | Set-Content (Join-Path $Dest "start_jupyter.bat") -Encoding ascii
Ok "env.ps1, env.bat, start_jupyter.bat dibuat di $Dest"

# 9. Verifikasi
if (-not $SkipVerify -and -not $NoTorch) {
    $env:COMP_ROOT = $Dest
    Info "Menjalankan verifikasi offline..."
    $rc = Run $vpy @((Join-Path $Dest "tools\03_verify_offline.py"))
    if ($rc -ne 0) { Warn "Verifikasi melaporkan masalah (kode $rc); baca keluaran di atas." }
}
Write-Host ""
Ok "Selesai. Mulai bekerja: buka terminal baru lalu jalankan  $Dest\start_jupyter.bat"
Stop-Transcript | Out-Null
