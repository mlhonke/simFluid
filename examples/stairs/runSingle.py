
import os
import numpy as np

os.chdir("cmake-build-debug")

# Trial Params
n_steps = 1000000

res = 30
# SimParams
grid_w = res
grid_h = res
grid_d = res
dx = 1.0/res
frac_cfl = 0.5
max_dt = 0.1
render_dt = 0.02
g = -9.8
# SimWaterParams
density = 1000.0
sft = 0.0001
nu = 0.01

paramNames = ["n_steps",
              "grid_w",
              "grid_h",
              "grid_d",
              "dx",
              "frac_cfl",
              "max_dt",
              "render_dt",
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
                frac_cfl,
                max_dt,
                render_dt,
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

os.system("./stairs" + args)
os.chdir("../screens")
videoName = tName
os.system("ffmpeg -i screen_%07d.png -c:v libx264 " + videoName + ".mp4")
os.system("rm *.png")
with open(tName + ".txt", 'w') as f:
    for param, paramName in zip(paramList, paramNames):
        f.write("%s: %s\n" % (paramName, param))
os.chdir("../cmake-build-debug")
