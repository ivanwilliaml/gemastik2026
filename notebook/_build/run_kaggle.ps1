# Push notebook ke Kaggle, tunggu selesai, unduh notebook dan file hasil kecil, lalu audit.
# Pemakaian: .\run_kaggle.ps1 -Ids 08,01 -Mode cepat -Parallel 2
param(
    [string[]]$Ids,
    [string]$Mode = "cepat",
    [int]$Parallel = 2,
    [int]$TimeoutSeconds = 7200
)

$Ids = @($Ids | ForEach-Object { '{0:00}' -f [int]$_ })
$root = Split-Path $PSScriptRoot -Parent
Set-Location $PSScriptRoot
python build.py --mode $Mode --only @Ids
if (-not $?) { exit 1 }

function Get-Task($id) {
    $json = python -c "import json,tasks; print(json.dumps([t for t in tasks.TASKS if t['id'].startswith('$id')][0]))"
    return $json | ConvertFrom-Json
}

function Save-Result($task) {
    $target = Join-Path $root $task.folder
    $hasil = Join-Path $target "hasil"
    New-Item -ItemType Directory -Force $hasil | Out-Null
    $tmp = Join-Path $env:TEMP ("kg_" + $task.slug)
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $tmp | Out-Null
    kaggle kernels output "ivanwllm/$($task.slug)" -p $tmp --file-pattern '^(hasil/.*\.(csv|json|png)|.*\.ipynb)$' -o -q | Out-Null
    $nb = Get-ChildItem $tmp -Filter *.ipynb | Select-Object -First 1
    if (-not $nb) { Write-Output "PERINGATAN: notebook berisi output tidak ada di output kernel $($task.slug)" }
    if ($nb) {
        $final = Join-Path $target "$($task.id).ipynb"
        Copy-Item $nb.FullName $final -Force
        python tidy.py $final
    }
    $small = Join-Path $tmp "hasil"
    if (Test-Path $small) { Copy-Item "$small\*" $hasil -Force -Recurse }
    $log = Get-ChildItem $tmp -Filter *.log | Select-Object -First 1
    if ($log) { Copy-Item $log.FullName (Join-Path $env:TEMP "gk\$($task.id)\run.log") -Force }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$queue = [System.Collections.ArrayList]@($Ids)
$running = @{}
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $Parallel) {
        $id = $queue[0]; $queue.RemoveAt(0)
        $task = Get-Task $id
        $out = kaggle kernels push -p (Join-Path $env:TEMP "gk\$($task.id)") -t $TimeoutSeconds 2>&1 | Out-String
        if ($out -match 'successfully pushed') {
            Write-Output "push $id"
            $running[$id] = $task
        } else {
            Write-Output "gagal push $id : $out"
        }
    }
    Start-Sleep -Seconds 45
    foreach ($id in @($running.Keys)) {
        $task = $running[$id]
        $status = kaggle kernels status "ivanwllm/$($task.slug)" 2>&1 | Out-String
        if ($status -notmatch 'RUNNING|QUEUED') {
            Write-Output "selesai $id : $($status.Trim())"
            Save-Result $task
            $running.Remove($id)
        }
    }
}
python audit.py --only @Ids
kaggle quota
