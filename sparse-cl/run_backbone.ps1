<#
.SYNOPSIS
Luoi backbone fine-tune: 6 cau hinh x {khong regularizer, EWC-DR lamda=100}.

.DESCRIPTION
Ban PowerShell cua run_backbone.sh, cho Windows khong co bash.

Ngan sach 100 epoch/task, patience 10.
LUU Y: bang ket qua chinh (92 run, ca ViT lan ResNet) chay o 100/20 va batch 256.
Luoi nay khac ca hai, nen phai so NOI BO voi dong frozen+Linear cua chinh no,
khong so tuyet doi voi 89.49 / 74.38.

train.py ghi JSON sau MOI task, nen dung giua chung van con ket qua den do -
xem 'tasks_done' va 'complete' trong file JSON.

.PARAMETER Backbone
vit hoac resnet.

.PARAMETER Configs
'a' (cau hinh 1,2,3), 'b' (4,5,6), 'all', hoac danh sach so nhu '1,2' / '5'.

.PARAMETER Regs
'none', 'ewc', hoac 'both'.

.PARAMETER BatchSize
0 = tu chon theo backbone: ResNet 128 (do duoc 5.8 GiB), ViT 64 (5.6 GiB o 64,
gap doi la ~11 GiB, sat tran 16 GiB nen khong nen).

.EXAMPLE
.\run_backbone.ps1 -Backbone resnet
.EXAMPLE
.\run_backbone.ps1 -Backbone vit -Configs '3,4' -Regs none
#>
param(
    [ValidateSet('vit', 'resnet')][string]$Backbone = 'resnet',
    [int]$Gpu = 0,
    [string]$Configs = 'all',
    [ValidateSet('none', 'ewc', 'both')][string]$Regs = 'both',
    [int]$BatchSize = 0,
    [int]$Seed = 1993,
    [string]$OutDir = './runs',
    [switch]$DryRun          # in lenh ra roi thoat, khong chay
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($Backbone -eq 'vit') {
    $model = 'vit_base_patch16_224'; $aug = 'vit'
    if ($BatchSize -eq 0) { $BatchSize = 64 }
} else {
    $model = 'resnet50'; $aug = 'resnet'
    if ($BatchSize -eq 0) { $BatchSize = 128 }
}

$common = @(
    '--model_name', $model, '--data_augmentation', $aug, '--gpu', $Gpu,
    '--freeze_backbone', 'False', '--backbone_lr', '1e-5',
    '--epochs', '100', '--early_stop_patience', '10',
    '--batch_size', $BatchSize, '--seed', $Seed, '--out_dir', $OutDir
)

$mlp    = @('--use_mlp', 'True', '--mlp_act', 'relu', '--mlp_hidden', '512')
$noMlp  = @('--use_mlp', 'False')
$frozen = @('--train_projection', 'False', '--projection_schedule', 'task0')
$learn  = @('--train_projection', 'True', '--projection_schedule', 'continual',
            '--projection_lr', '5e-3')

# Cung 6 cau hinh cua bang ket qua chinh. Khac mot diem: khi backbone hoc duoc
# thi MOI cau hinh deu co tham so troi qua cac task, nen o EWC khong con o n/a.
$cfg = [ordered]@{
    '1_none_linear'   = @('--expand_dim', '0') + $noMlp
    '2_none_mlp'      = @('--expand_dim', '0') + $mlp
    '3_frozen_linear' = $frozen + $noMlp
    '4_frozen_mlp'    = $frozen + $mlp
    '5_learn_linear'  = $learn + $noMlp
    '6_learn_mlp'     = $learn + $mlp
}

switch ($Configs) {
    'a'   { $keys = @('1_none_linear', '2_none_mlp', '3_frozen_linear') }
    'b'   { $keys = @('4_frozen_mlp', '5_learn_linear', '6_learn_mlp') }
    'all' { $keys = @($cfg.Keys) }
    default {
        $keys = @()
        foreach ($n in $Configs -split ',') {
            $k = @($cfg.Keys) | Where-Object { $_ -like "$($n.Trim())_*" }
            if (-not $k) { throw "cau hinh khong hop le: '$n' (chon 1..6, a, b hoac all)" }
            $keys += $k
        }
    }
}

# lamda=100 lay tu sweep {1,10,100,1000,10000} tren ViT backbone dong bang.
switch ($Regs) {
    'none' { $regList = @( , @('--cl_reg', 'none')) }
    'ewc'  { $regList = @( , @('--cl_reg', 'ewc_dr', '--lamda', '100')) }
    'both' { $regList = @(@('--cl_reg', 'none'), @('--cl_reg', 'ewc_dr', '--lamda', '100')) }
}

$py = (Get-Command python).Source
Write-Output "python   : $py"
Write-Output "backbone : $model | batch $BatchSize | gpu $Gpu"
Write-Output "cau hinh : $($keys -join ', ')"
Write-Output "so run   : $($keys.Count * $regList.Count)"
Write-Output ''

$failed = @()
$t0 = Get-Date
foreach ($k in $keys) {
    foreach ($reg in $regList) {
        Write-Output "===== $Backbone | $k | $($reg -join ' ') | gpu $Gpu ====="
        if ($DryRun) {
            Write-Output "  python -u train.py $($common -join ' ') $($cfg[$k] -join ' ') $($reg -join ' ')"
            continue
        }
        # -u: khong dem stdout, neu khong log trong hang chuc phut du dang chay
        & $py -u train.py @common @($cfg[$k]) @reg
        if ($LASTEXITCODE -ne 0) {
            Write-Output "LOI: $k $($reg -join ' ') -> ma thoat $LASTEXITCODE"
            $failed += "$k $($reg -join ' ')"
        }
    }
}

$h = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
Write-Output ''
Write-Output "XONG: $Backbone | cau hinh $Configs | reg $Regs | $h gio"
if ($failed.Count) { Write-Output "That bai: $($failed -join ' ; ')" }
