param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)

$here = Get-Location

try {
    $ptFile = Get-Item $InputFile
    $ptFullPath = $ptFile.FullName
    $onnxFullPath = $ptFile.DirectoryName + "/" + $ptFile.BaseName + ".onnx"
    $tfjsDir = $ptFile.DirectoryName + "/onnx2tfjs/" + $ptFile.BaseName + ".onnx-tf-tfjs-uint8"

    Write-Host "PT model path: $ptFullPath"
    Write-Host "ONNX model path: $onnxFullPath"

    # Convert PyTorch to ONNX
    Write-Host "Converting PyTorch model to ONNX..."
    Set-Location pt2onnx

    Invoke-Expression "uv sync"
    Invoke-Expression "uv run yolo export model=${ptFullPath} format=onnx opset=12"

    # Convert ONNX to TensorFlow.js
    Write-Host "Converting ONNX model to TensorFlow.js..."
    Set-Location ../onnx2tfjs
    Invoke-Expression "uv sync"
    Invoke-Expression "uv run ./onnx2tfjs.py $onnxFullPath"
    Write-Host "Conversion completed successfully. Output files are in ${tfjsDir}/"
}
catch {
    Write-Error "An error occurred during conversion: $_"
    exit 1
}
finally {
    Set-Location $here
}
