import bpy
import numpy
import mathutils
import sys
import numpy as np
import shutil
import os
# from PIL import Image

argv = sys.argv
argv = argv[argv.index("--args") + 1:] # read all args: path_to_camera, path_to_mesh, path_to_hair_data

def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = list(cycles_preferences.devices)[:2]

    activated_gpus = []

    for device in devices:
        device.use = True
        activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True

    return activated_gpus

enable_gpus()

# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations: 
#   - Oxford's visual geometry group matlab toolbox 
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)

    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K @ sg
    R = sg @ R
    # det(R) negative, just invert; the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
        R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:,-1])[0]
    T = -R @ C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    
    return r, q

# Creates a blender camera consistent with a given 3x4 computer vision P matrix
# Run this in Object Mode
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def get_blender_camera_from_3x4_P(P, scale, name):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))
    K[0,2] = K[1,2]

    scene = bpy.context.scene
    
    sensor_width_in_mm = K[0,0]*1080 / (K[1,1]*K[1,2])#K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
    sensor_height_in_mm = 1# sensor_width_in_mm / K[0, 0]
    
    
    resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center 

    
    print('sensor width', sensor_width_in_mm,resolution_x_in_px, resolution_y_in_px, sensor_height_in_mm, K)
    
    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
#     s_u = 1080 / sensor_width_in_mm
#     s_v = 1920 / sensor_height_in_mm
    # TODO include aspect ratio
#     f_in_mm = K[0,0] / s_u sensor_width_in_mm / resolution_x_in_px
#     f_in_mm = K[0,0] * sensor_width_in_mm / resolution_x_in_px  # Focal length in mm
    f_in_mm = K[0,0] / s_u

    print('fin mm', f_in_mm,s_u, s_v)
    # recover original resolution
    scene.render.resolution_x = int(resolution_x_in_px) // scale
    scene.render.resolution_y = int(resolution_y_in_px) // scale
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = mathutils.Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation =  mathutils.Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = -R_cv2world @ T_world2cv

    # create a new camera
    bpy.ops.object.add(
        type='CAMERA',
        location=location)
    ob = bpy.context.object
    ob.name = name
    cam = ob.data
    cam.name = name

    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm 
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm
    cam.sensor_height = sensor_height_in_mm
    
    obj_mat = mathutils.Matrix.Translation(location) @ rotation.to_4x4()
    
    tmp = numpy.array(obj_mat)
    
    ob.matrix_world = mathutils.Matrix(tmp)

    # Display
    cam.show_name = True

    return ob    

def set_material(obj, material):
    if len(obj.material_slots) < 1:
        obj.data.materials.append(material) 
    else:
        obj.material_slots[obj.active_material_index].material = material

# Load camera
cameras = np.load(argv[0])[:, :3, :4]
camera_obs = []
for i in range(cameras.shape[0]):
    print(f'loading camera {i}')
    camera_obs.append(get_blender_camera_from_3x4_P(cameras[i], 1, str(i)))

# Load head mesh
print(f'Loading head mesh from {argv[1]}')
bpy.ops.import_mesh.ply(filepath=argv[1])
head_object_name = argv[1].split('/')[-1].split('.')[0]
print(head_object_name)
set_material(bpy.data.objects[head_object_name], bpy.data.materials['Main'])

if len(argv) >= 3:
    # Load hair
    hair_blocks = 4
    hair = np.load(argv[2])
    n_strands = 25000
    print(hair.shape)
    hair = hair[np.random.choice(len(hair), n_strands, replace=False)]

    color_scheme = [
        (0.125, 0.5, 0.0, 1.0),
        (0.5, 0.0, 0.0, 1.0),
        (0.125, 0.0, 0.5, 1.0),
        (0.0, 0.5, 0.5, 1.0)
    ]

    def create_hair(name: str, block_id: int):

        def create_points(curveData, coords, index=0):
            polyline = curveData.splines.new('POLY')
            polyline.points.add(len(coords)-1)
            for i, coord in enumerate(coords):
                x, y, z = coord
                polyline.points[index].co = (x, y, z, i)
                index += 1

        hair_sample = hair[block_id * (n_strands // hair_blocks): (block_id + 1) * (n_strands // hair_blocks)]
        
        curveData = bpy.data.curves.new('hair', type='CURVE')
        curveData.dimensions = '3D'
        curveData.resolution_u = 1

        for i in range(len(hair_sample)):
            index = 0
            create_points(curveData, hair_sample[i], index=index)

        curveOB = bpy.data.objects.new(name, curveData)
        bpy.data.scenes[0].collection.objects.link(curveOB)
        
        return bpy.data.objects[name]


    def create_hair_material(name):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        output = nodes.new(type='ShaderNodeOutputMaterial')
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')

        links.new(shader.outputs[0], output.inputs[0])
        
        return bpy.data.materials[name]


    hair_objects = [create_hair(f'Hair {i}', i) for i in range(hair_blocks)]
    hair_materials = [create_hair_material(f'Hair Material {i}') for i in range(hair_blocks)]

    for i in range(4):
        hair_objects[i].rotation_euler[0] = - np.pi / 2

        #set_material(hair_objects[i], bpy.data.materials['Main'])
        set_material(hair_objects[i], hair_materials[i])
        
        #hair_materials[i].node_tree.nodes["RGB"].outputs[0].default_value = color_scheme[i]
        hair_materials[i].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color_scheme[i]
        hair_materials[i].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.0

        hair_objects[i].data.bevel_depth = 0.0012 

bpy.data.objects.remove(bpy.data.objects['placeholder'])


# bpy.data.scenes['Scene'].node_tree.nodes['File Output'].base_path = '.'
for scene in bpy.data.scenes:
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1920

exp_path = argv[3]
os.makedirs(exp_path, exist_ok=True)

idx_start = int(argv[5])
idx_offset = int(argv[6])

for camera_idx, camera_ob in enumerate(camera_obs):
    camera_idx = camera_idx * idx_offset + idx_start
    print(f'rendering frame {camera_idx}')
    # Make this the current camera
    scene.camera = camera_ob
    bpy.context.scene.cycles.samples = int(argv[4])
    bpy.context.view_layer.update()
    bpy.ops.render.render()

    shutil.move('image0000.png', f'{exp_path}/%06d.png' % camera_idx)
