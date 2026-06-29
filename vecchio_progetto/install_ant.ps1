$ErrorActionPreference = 'Stop'
$base = 'https://downloads.apache.org/ant/binaries/'
Write-Host "Querying $base for available Ant binaries..."
$resp = Invoke-WebRequest -Uri $base -UseBasicParsing
$links = $resp.Links | Where-Object { $_.href -match 'apache-ant-.*-bin.zip$' } | Select-Object -ExpandProperty href
if (-not $links) { Write-Error "No Ant binary links found at $base"; exit 1 }
$latest = $links | Sort-Object -Descending | Select-Object -First 1
$url = $base + $latest
$dest = Join-Path $env:USERPROFILE $latest
Write-Host "Downloading $url to $dest"
Invoke-WebRequest -Uri $url -OutFile $dest
$extractDir = Join-Path $env:USERPROFILE 'apache-ant'
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
Expand-Archive -Path $dest -DestinationPath $extractDir
$antPath = Get-ChildItem $extractDir | Where-Object {$_.PSIsContainer} | Select-Object -First 1
if (-not $antPath) { Write-Error "Extraction failed or no folder found in $extractDir"; exit 1 }
$antFull = $antPath.FullName
Write-Host "Installed to $antFull"
[Environment]::SetEnvironmentVariable('ANT_HOME', $antFull, 'User')
$uPath = [Environment]::GetEnvironmentVariable('Path','User')
$antBin = Join-Path $antFull 'bin'
if ($uPath -notlike "*$antBin*") { [Environment]::SetEnvironmentVariable('Path', $uPath + ';' + $antBin, 'User') }
$env:ANT_HOME = $antFull
$env:Path = $env:Path + ';' + $antBin
Write-Host "ANT_HOME set to $antFull"
Write-Host "Verifying installation..."
ant -version
Write-Host "Done. You may need to open a new terminal to pick up PATH changes."