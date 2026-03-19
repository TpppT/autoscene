param(
    [ValidateSet("svg", "png")]
    [string[]]$Formats = @("svg", "png")
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$docsRoot = Join-Path $repoRoot "docs"
$plantUmlJar = Join-Path $repoRoot "tools\plantuml.jar"

if (-not (Test-Path $plantUmlJar)) {
    throw "PlantUML jar not found: $plantUmlJar"
}

$javaCommand = Get-Command java -ErrorAction SilentlyContinue
if ($null -eq $javaCommand) {
    throw "Java was not found on PATH."
}

$diagramFiles = Get-ChildItem -Path $docsRoot -Recurse -Filter *.puml | Sort-Object FullName
if ($diagramFiles.Count -eq 0) {
    throw "No PlantUML files were found under $docsRoot"
}

$diagramPaths = $diagramFiles.FullName

Write-Host "Rendering $($diagramFiles.Count) diagrams from $docsRoot"

foreach ($format in $Formats) {
    Write-Host " -> generating $format"
    & $javaCommand.Source -jar $plantUmlJar "-t$format" @diagramPaths
}

Write-Host "Diagram render complete."
