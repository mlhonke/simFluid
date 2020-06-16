
import os
import numpy as np

os.chdir("cmake-build-debug")

# Trial Params
n_steps = 1000000

# SimParams
grid_w = 60
grid_h = 60
grid_d = 60
dx = 1.0/60.0
g = -9.8
# SimWaterParams
density = 1000.0
sft = 0.0000
nu = 0.0

paramNames = ["n_steps",
              "grid_w",
              "grid_h",
              "grid_d",
              "dx",
              "density",
              "sft",
              "nu",
              "g"]


def makeParamList():
    paramListOut = [
                n_steps,
                grid_w,
                grid_h,
                grid_d,
                dx,
                density,
                sft,
                nu,
                g]
    return paramListOut


def paramListToArgs(paramListIn):
    argsOut = ""
    for param in paramList:
        argsOut = argsOut + " "
        argsOut = argsOut + str(param)
    return argsOut


#sft_range = np.arange(0, 0.010, 0.001)


tName = "trial_single"

# Edit the parameters here for each trial

paramList = makeParamList()
args = paramListToArgs(paramList)

print(args)

os.system("./freefall" + args)
os.chdir("../screens")
videoName = tName
os.system("ffmpeg -i screen_%07d.png -c:v libx264 " + videoName + ".mp4")
os.system("rm *.png")
with open(tName + ".txt", 'w') as f:
    for param, paramName in zip(paramList, paramNames):
        f.write("%s: %s\n" % (paramName, param))
os.chdir("../cmake-build-debug")
