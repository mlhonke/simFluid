import subprocess
from pathlib import Path
from time import sleep
import os
import signal

render_time = 1
mesh_id = 0
mesh_directory = "screens/" # Where are the mesh files located to render?
output_dir = "renders/" # Where to save rendering related files and frames.

def makeVideoFromFrames():
    os.system("ffmpeg -i " + output_dir + "frame_%07d.png -c:v libx264 -crf 0 -preset medium " + "render_output" + ".mp4")
    os.system("ffmpeg -i " + output_dir + "frame_%07d_noisy.png -c:v libx264 -crf 0 -preset medium " + "render_output_noisy" + ".mp4")

while True:
    fluid_mesh = mesh_directory + "mesh_%07d.ply" % mesh_id
    
    if Path(fluid_mesh).is_file() is not True: # max number for testing
        print("Mesh file " + fluid_mesh + " not found. Rendering finished")
        break
    
    renderProc = subprocess.Popen(["python3 makeRaytracedVideo.py " + str(mesh_id) + " " + str(render_time)], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    print("Rendering process for frame " + str(mesh_id) + " started.")
    try:
        renderProc.wait(timeout = 2*render_time+5)
    except subprocess.TimeoutExpired:
        print("==================== RENDER PROCESS ERROR ====================")
        print("Rendering process for frame " + str(mesh_id) + " hung. Killing this frame's process.")
        print("==============================================================")
        os.killpg(os.getpgid(renderProc.pid), signal.SIGTERM)
        
    mesh_id += 1

makeVideoFromFrames()
