from time import sleep
import array
import os
from pathlib import Path
import sys
# Put your luxcore python module path here.
sys.path.append('/home/graphics/Programs/LuxCore-opencl/')
#sys.path.append('/home/graphics/Programs/LuxCore/build/lib/')
import pyluxcore
print('Using luxcorerender version ' + pyluxcore.Version())

mesh_directory = "screens/" # Where are the mesh files located to render?
output_dir = "renders/" # Where to save rendering related files and frames.

def build_scene(fluid_mesh):
    scene = pyluxcore.Scene()
    props = pyluxcore.Properties()

    props.SetFromString("""
        scene.camera.type = "perspective"
        scene.camera.lookat.orig = 0.5 -2.0 1.0
        scene.camera.lookat.target = 0.5 0.5 0.4
        scene.camera.fieldofview = 35
        ################################################################################
        scene.materials.fluid.type = glass
        #scene.materials.fluid.kd = 1. 0.824176 0.549451
        scene.materials.fluid.kr = 1.0 1.0 1.0
        scene.materials.fluid.kt = 1.0 1.0 1.0
        scene.materials.fluid.interiorior = 1.333
        scene.materials.fluid.exteriorior = 1.0
        scene.materials.fluid.photongi.enable = 1
        ################################################################################
        scene.textures.table.type = blender_noise
        scene.materials.slab.type = matte
        #scene.materials.slab.kd = table
        scene.materials.slab.kd = 0.8 0.64 0.45
        ################################################################################
        #scene.materials.container.type = roughglass
        #scene.materials.container.kr = 0.9 0.9 0.9
        #scene.materials.container.kt = 0.9 0.9 0.9
        #scene.materials.container.interiorior = 1.5
        #scene.materials.container.exteriorior = 1.0
        #scene.materials.container.uroughness = 0.5
        #scene.materials.container.vroughness = 0.5
        scene.materials.container.type = matte
        scene.materials.container.kd = 1.0 1.0 1.0
        ################################################################################
        #scene.lights.l1.type = sky2
        #scene.lights.l1.gain = 0.0003 0.0003 0.0003
        #scene.lights.l1.dir = 0.0 0.0 1.0
        #scene.lights.l1.relsize = 1.0
        #scene.lights.l1.turbidity = 2.0
        #scene.lights.l1.ground.enable = 0
        
        scene.lights.l2.type = point
        scene.lights.l2.position = 0.5 0.5 1.0
        scene.lights.l2.gain = 0.7 0.7 0.7
        
        scene.lights.l3.type = point
        scene.lights.l3.position = 0.0 0.5 2.0
        scene.lights.l3.gain = 0.99 0.99 0.99
        ################################################################################
        scene.objects.slab.material = slab
        scene.objects.slab.ply = slab.ply
        #scene.objects.container.material = container
        #scene.objects.container.ply = container_tri.ply
        #scene.objects.stairs.material = container
        #scene.objects.stairs.ply = stairs.ply
        scene.objects.solid.material = container
        scene.objects.solid.ply = solid.ply
        ################################################################################
        """)
    scene.Parse(props)
    
    props = pyluxcore.Properties()
    props.Set(pyluxcore.Property("scene.objects.fluid.material", "fluid"))
    props.Set(pyluxcore.Property("scene.objects.fluid.ply", fluid_mesh))
    scene.Parse(props)
  
    return scene


def build_session(scene, output_name):
    props = pyluxcore.Properties()
    props.SetFromString("""
        opencl.cpu.use = 0
        # Use all GPU devices we can find
        opencl.gpu.use = 1
        # You can use this setting to specify exactly which OpenCL devices to use
        #opencl.devices.select = "1"

        renderengine.type = "PATHCPU"
        sampler.type = "SOBOL"
        #renderengine.type = "TILEPATHOCL"
        #sampler.type = "TILEPATHSAMPLER"
        #pathocl.pixelatomics.enable = 1
        renderengine.seed = 1
        path.pathdepth.total = 17
        path.pathdepth.diffuse = 10
        path.pathdepth.glossy = 17
        path.pathdepth.specular = 16

        path.photongi.sampler.type = METROPOLIS
        path.photongi.photon.maxcount = 10000000
        path.photongi.photon.maxdepth = 16
        path.photongi.photon.time.start = 0.0
        path.photongi.photon.time.end = 1.0
        #path.photongi.indirect.enabled = 1
        #path.photongi.indirect.maxsize = 100000
        #path.photongi.indirect.haltthreshold = 0.05
        #path.photongi.indirect.lookup.radius = 0.15
        #path.photongi.indirect.glossinessusagethreshold = 0.0
        #path.photongi.indirect.usagethresholdscale = 0.0
        #path.photongi.indirect.filter.radiusscale = 4.0
        #path.photongi.caustic.enabled = 0
        #path.photongi.caustic.maxsize = 10000
        #path.photongi.caustic.lookup.radius = 0.15
        #path.photongi.debug.type = showindirect
        #path.photongi.debug.type = showcaustic
        #path.photongi.debug.type = showindirectpathmix
        path.photongi.persistent.file = cornell.pgi
                
        film.width = 768
        film.height = 768
        
        film.imagepipelines.0.0.type = "NOP"
        film.imagepipelines.0.1.type = "TONEMAP_LUXLINEAR"
        film.imagepipelines.0.1.fstop = 4
        film.imagepipelines.0.1.exposure = 100
        film.imagepipelines.0.1.sensitivity = 150
        film.imagepipelines.0.2.type = "GAMMA_CORRECTION"
        film.imagepipelines.0.2.value = 2.2000000000000002
        
        film.imagepipelines.1.0.type = INTEL_OIDN
        film.imagepipelines.1.1.type = "TONEMAP_LUXLINEAR"
        film.imagepipelines.1.1.fstop = 4
        film.imagepipelines.1.1.exposure = 100
        film.imagepipelines.1.1.sensitivity = 150
        film.imagepipelines.1.2.type = "GAMMA_CORRECTION"
        film.imagepipelines.1.2.value = 2.2000000000000002
        
        #film.imagepipelines.0.1.type = GAMMA_CORRECTION
        #film.imagepipelines.0.1.value = 7.0
        #film.imagepipelines.0.0.type = BCD_DENOISER
        #film.imagepipelines.0.1.type = GAMMA_CORRECTION
        #film.imagepipelines.0.1.value = 2.2
        
        film.filter.type = "BLACKMANHARRIS"
        film.filter.width = 1.5
        film.outputs.0.type = "RGB_IMAGEPIPELINE"
        film.outputs.0.index = 0
        film.outputs.0.filename = """ + output_name + "_noisy.png" + """
        film.outputs.1.type = RGB_IMAGEPIPELINE
        film.outputs.1.index = 1
        film.outputs.1.filename = """ + output_name + ".png" + """
        film.outputs.2.type = ALBEDO
        film.outputs.2.filename = """ + output_name + "_" + "ALBEDO.png" + """
        film.outputs.3.type = AVG_SHADING_NORMAL
        film.outputs.3.filename = """ + output_name + "_" + "AVG_SHADING_NORMAL.png" + """
        scene.file = "freefall.scn"
        """)

    renderconfig = pyluxcore.RenderConfig(props, scene)
    session = pyluxcore.RenderSession(renderconfig)
    return session


def main():
    print(sys.argv)
    mesh_id = int(sys.argv[1])
    render_time = int(sys.argv[2])
    # You have to init pyluxcore before you call anything else

    pyluxcore.Init()
        
    fluid_mesh = mesh_directory + "mesh_%07d.ply" % mesh_id
        
    if Path(fluid_mesh).is_file() is not True: # max number for testing
        print("Mesh file " + fluid_mesh + " not found. Rendering finished")
    else:
        print("Rendering mesh " + fluid_mesh)
            
        scene = build_scene(fluid_mesh)
        output_name = output_dir + "frame_%07d" % mesh_id
        session = build_session(scene, output_name)
        session.Start()
        sleep(render_time)
        session.GetFilm().SaveOutputs()
        session.Stop()
        
        
if __name__ == "__main__":
    main()
