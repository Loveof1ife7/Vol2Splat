from paraview import servermanager

presets = servermanager.vtkSMTransferFunctionPresets()
n = presets.GetNumberOfPresets()

with open("pvpython_colormaps.txt", "w", encoding="utf-8") as f:
    for i in range(n):
        f.write(f"{i}\t{presets.GetPresetName(i)}\n")

print("saved to pvpython_colormaps.txt")