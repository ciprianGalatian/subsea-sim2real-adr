import blenderproc as bproc
import numpy as np
import os
import random
import shutil
import string
import glob
import json
import argparse
import itertools
import bpy 
from datetime import datetime 

def parse_args():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_PBR = os.path.join(BASE_DIR, "data", "cc_materials")
    
    parser = argparse.ArgumentParser(description="Domain randomization generator")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--num_meshes_per_class", type=int, default=1)
    parser.add_argument("--num_distractors", type=int, default=0)
    parser.add_argument("--disable_deterministic", action="store_true")
    parser.add_argument("--multi_class_chance", type=float, default=0.0)
    parser.add_argument("--multi_class_max", type=int, default=5)
    parser.add_argument("--min_mesh_distance", type=float, default=0.25)
    parser.add_argument("--enable_text_distractors", action="store_true")
    parser.add_argument("--num_text_chars", type=int, default=10)
    parser.add_argument("--disable_strict_mesh_filter", action="store_true")
    parser.add_argument("--texture_mode", type=str, choices=["neon", "hsv", "pbr"], default="neon")
    parser.add_argument("--pbr_materials_dir", type=str, default=DEFAULT_PBR)
    parser.add_argument("--disable_shadows", action="store_true")
    parser.add_argument("--num_light_sources", type=int, default=3)
    parser.add_argument("--light_color_mode", type=str, choices=["Random", "Deep_Blue_Green"], default="Random")
    parser.add_argument("--enable_motion_blur", action="store_true")
    parser.add_argument("--motion_blur_amount", type=float, default=0.05)
    parser.add_argument("--target_scale_min", type=float, default=0.35)
    parser.add_argument("--target_scale_max", type=float, default=0.8)
    parser.add_argument("--camera_radius_min", type=float, default=1.5)
    parser.add_argument("--camera_radius_max", type=float, default=4.0)
    parser.add_argument("--camera_elevation_min", type=float, default=10.0)
    parser.add_argument("--camera_elevation_max", type=float, default=85.0)
    parser.add_argument("--enable_extreme_offsets", action="store_true")
    parser.add_argument("--negative_sample_rate", type=float, default=0.05)

    return parser.parse_args()

def main():
    args = parse_args()
    
    NUM_IMAGES = args.num_images
    NUM_MESHES_PER_CLASS = args.num_meshes_per_class
    NUM_DISTRACTORS = args.num_distractors
    ENABLE_DETERMINISTIC = not args.disable_deterministic
    MULTI_CLASS_CHANCE = args.multi_class_chance
    MULTI_CLASS_MAX = args.multi_class_max
    MIN_MESH_DISTANCE = args.min_mesh_distance
    ENABLE_TEXT_DISTRACTORS = args.enable_text_distractors
    NUM_TEXT_CHARS = args.num_text_chars
    ENABLE_STRICT_MESH_FILTER = not args.disable_strict_mesh_filter
    ENABLE_TEXTURE_RANDOMIZATION = (args.texture_mode == "neon")
    ENABLE_HSV_HUE_SHIFT = (args.texture_mode == "hsv")
    ENABLE_PBR_RANDOMIZATION = (args.texture_mode == "pbr")
    PBR_MATERIALS_DIR = args.pbr_materials_dir
    ENABLE_SHADOWS = not args.disable_shadows
    NUM_LIGHT_SOURCES = args.num_light_sources
    LIGHT_COLOR_MODE = args.light_color_mode
    ENABLE_MOTION_BLUR = args.enable_motion_blur
    MOTION_BLUR_AMOUNT = args.motion_blur_amount
    TARGET_SCALE_RANGE = [args.target_scale_min, args.target_scale_max]
    CAMERA_RADIUS_RANGE = [args.camera_radius_min, args.camera_radius_max]
    CAMERA_ELEVATION_RANGE = [args.camera_elevation_min, args.camera_elevation_max]
    ENABLE_EXTREME_OFFSETS = args.enable_extreme_offsets
    NEGATIVE_SAMPLE_RATE = args.negative_sample_rate

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"blenderproc_output_{run_id}"

    bproc.init()

    pbr_materials = []
    organic_pbr = []
    artificial_pbr = []
    
    if ENABLE_PBR_RANDOMIZATION:
        print(f"Loading PBR Materials from {PBR_MATERIALS_DIR}...")
        if not os.path.exists(PBR_MATERIALS_DIR):
            print("\n" + "!"*60)
            print(" PBR DIRECTORY NOT FOUND")
            print("!"*60)
            print(" To use PBR Randomization, you must first download the materials.")
            print(f" Open your terminal and run the following command:")
            print(f"blenderproc download cc_textures {PBR_MATERIALS_DIR}")
            print("!"*60 + "\n")
            sys.exit(1)
            
        pbr_materials = bproc.loader.load_ccmaterials(PBR_MATERIALS_DIR)
        print(f"Successfully loaded {len(pbr_materials)} total PBR materials.")
        
        organic_keywords = ['rock', 'ground', 'wood', 'bark', 'sand', 'gravel', 'mud', 'moss', 'leaves', 'organic', 'stone', 'soil']
        
        for mat in pbr_materials:
            mat_name = mat.get_name().lower()
            if any(kw in mat_name for kw in organic_keywords):
                organic_pbr.append(mat)
            else:
                artificial_pbr.append(mat)

    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    BACKGROUNDS_DIR = os.path.join(BASE_PATH, "backgrounds")
    MESH_BASE_DIR = os.path.join(BASE_PATH, "meshes")
    
    OUTPUT_DIR = os.path.join(BASE_PATH, dataset_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Creating new unique dataset folder at: {OUTPUT_DIR}")

    config_path = os.path.join(OUTPUT_DIR, "dataset_config.txt")
    with open(config_path, "w") as config_file:
        config_file.write("=== DATASET GENERATION CONFIGURATION ===\n")
        config_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        config_file.write(f"NUM_IMAGES: {NUM_IMAGES}\n")
        config_file.write(f"NUM_MESHES_PER_CLASS: {NUM_MESHES_PER_CLASS}\n")
        config_file.write(f"NUM_DISTRACTORS: {NUM_DISTRACTORS}\n")
        config_file.write(f"MULTI_CLASS_CHANCE: {MULTI_CLASS_CHANCE}\n")
        config_file.write(f"MULTI_CLASS_MAX: {MULTI_CLASS_MAX}\n")
        config_file.write(f"MIN_MESH_DISTANCE: {MIN_MESH_DISTANCE}\n")
        config_file.write(f"ENABLE_DETERMINISTIC: {ENABLE_DETERMINISTIC}\n")
        config_file.write(f"ENABLE_TEXT_DISTRACTORS: {ENABLE_TEXT_DISTRACTORS}\n")
        config_file.write(f"NUM_TEXT_CHARS: {NUM_TEXT_CHARS}\n")
        config_file.write(f"ENABLE_STRICT_MESH_FILTER: {ENABLE_STRICT_MESH_FILTER}\n")
        config_file.write(f"ENABLE_TEXTURE_RANDOMIZATION: {ENABLE_TEXTURE_RANDOMIZATION}\n")
        config_file.write(f"ENABLE_HSV_HUE_SHIFT: {ENABLE_HSV_HUE_SHIFT}\n")
        config_file.write(f"ENABLE_PBR_RANDOMIZATION: {ENABLE_PBR_RANDOMIZATION}\n")
        config_file.write(f"ENABLE_SHADOWS: {ENABLE_SHADOWS}\n")
        config_file.write(f"NUM_LIGHT_SOURCES: {NUM_LIGHT_SOURCES}\n")
        config_file.write(f"LIGHT_COLOR_MODE: {LIGHT_COLOR_MODE}\n")
        config_file.write(f"ENABLE_MOTION_BLUR: {ENABLE_MOTION_BLUR}\n")
        config_file.write(f"MOTION_BLUR_AMOUNT: {MOTION_BLUR_AMOUNT}\n")
        config_file.write(f"TARGET_SCALE_RANGE: {TARGET_SCALE_RANGE}\n")
        config_file.write(f"CAMERA_RADIUS_RANGE: {CAMERA_RADIUS_RANGE}\n")
        config_file.write(f"CAMERA_ELEVATION_RANGE: {CAMERA_ELEVATION_RANGE}\n")
        config_file.write(f"ENABLE_EXTREME_OFFSETS: {ENABLE_EXTREME_OFFSETS}\n")
        config_file.write(f"NEGATIVE_SAMPLE_RATE: {NEGATIVE_SAMPLE_RATE}\n")
        config_file.write("========================================\n")
    print(f"Saved generation configuration to dataset_config.txt")

    print("Pre-processing 3D Objects into pool...")
    temp_gltf = os.path.join(BASE_PATH, "temp_joined_mesh.glb")
    
    valid_classes = {}  
    class_to_id = {}
    
    if not os.path.exists(MESH_BASE_DIR):
        raise ValueError(f"Mesh directory not found: {MESH_BASE_DIR}")

    for idx, class_name in enumerate(sorted(os.listdir(MESH_BASE_DIR))):
        class_path = os.path.join(MESH_BASE_DIR, class_name)
        if os.path.isdir(class_path):
            glbs = glob.glob(os.path.join(class_path, "*.glb"))
            if not glbs:
                continue 
                
            glbs = sorted(glbs)[:NUM_MESHES_PER_CLASS]
                
            cat_id = idx + 1
            class_to_id[class_name] = cat_id
            valid_classes[class_name] = []
            
            for glb_file in glbs:
                existing_objs = set(bpy.context.scene.objects)
                
                bpy.ops.import_scene.gltf(filepath=glb_file)
                
                new_objs = [obj for obj in bpy.context.scene.objects if obj not in existing_objs]
                imported_meshes = [obj for obj in new_objs if obj.type == 'MESH']
                
                valid_meshes = []
                if ENABLE_STRICT_MESH_FILTER:
                    for m in imported_meshes:
                        if m.hide_render or m.hide_viewport:
                            continue
                            
                        is_invisible = False
                        if m.data.materials:
                            for mat in m.data.materials:
                                if mat and mat.node_tree:
                                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                                    if bsdf:
                                        alpha_input = bsdf.inputs.get("Alpha")
                                        if alpha_input and not alpha_input.links:
                                            if alpha_input.default_value < 0.05:
                                                is_invisible = True
                        if is_invisible:
                            continue
                            
                        valid_meshes.append(m)
                else:
                    valid_meshes = imported_meshes
                
                if len(valid_meshes) > 0:
                    bpy.ops.object.select_all(action='DESELECT')
                    for m in valid_meshes:
                        m.select_set(True)
                    bpy.context.view_layer.objects.active = valid_meshes[0]
                    if len(valid_meshes) > 1:
                        bpy.ops.object.join()
                    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                    
                    bpy.ops.export_scene.gltf(filepath=temp_gltf, use_selection=True)
                
                bpy.ops.object.select_all(action='DESELECT')
                for obj in new_objs:
                    try:
                        if obj.name in bpy.context.scene.objects:
                            obj.select_set(True)
                    except Exception:
                        pass
                bpy.ops.object.delete()
                
                if len(valid_meshes) > 0 and os.path.exists(temp_gltf):
                    loaded_objs = bproc.loader.load_obj(temp_gltf)
                    for obj in loaded_objs:
                        obj.set_cp("category_id", cat_id)
                        obj.set_location([1000, 1000, 1000]) 
                        valid_classes[class_name].append(obj)

    with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
        for class_name, cat_id in sorted(class_to_id.items(), key=lambda item: item[1]):
            f.write(f"{cat_id}: {class_name}\n")
            print(f"  [ID {cat_id}] Registered: {class_name} ({len(valid_classes[class_name])} pool meshes)")

    bg_plane = bproc.object.create_primitive('PLANE', scale=[15, 15, 1]) 
    bg_plane.set_location([0, 0, -1]) 

    valid_exts = {'.jpg', '.jpeg', '.png'}
    bg_images = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR) if os.path.splitext(f)[1].lower() in valid_exts]

    if not bg_images:
        raise ValueError(f"No background images found in {BACKGROUNDS_DIR}")

    lights = [bproc.types.Light() for _ in range(NUM_LIGHT_SOURCES)]
    for light in lights:
        light.set_type("POINT")

    distractors = []
    for j in range(NUM_DISTRACTORS):
        shape_type = random.choice(['CUBE', 'SPHERE', 'CYLINDER', 'CONE'])
        distractor = bproc.object.create_primitive(shape_type, scale=[0.1, 0.1, 0.1])
        dist_mat = bproc.material.create(f"distractor_mat_{j}")
        distractor.replace_materials(dist_mat)
        distractors.append(distractor)

    text_distractors = []
    pool_size = 10 
    if ENABLE_TEXT_DISTRACTORS:
        print(f"Generating 3D Text Distractor Pool (Length: {NUM_TEXT_CHARS})...")
        for k in range(pool_size): 
            random_text = ''.join(random.choices(string.ascii_uppercase + string.digits + " :-_", k=NUM_TEXT_CHARS))
            
            bpy.ops.object.text_add(location=(0, 0, 0))
            text_bpy = bpy.context.object
            text_bpy.data.body = random_text
            text_bpy.data.extrude = 0.02 
            
            bpy.context.view_layer.objects.active = text_bpy
            bpy.ops.object.convert(target='MESH')
            
            text_bproc = bproc.types.MeshObject(text_bpy)
            text_mat = bproc.material.create(f"telemetry_mat_{k}")
            text_mat.make_emissive(emission_strength=random.uniform(3.0, 8.0), emission_color=[1.0, 1.0, 1.0, 1.0])
            text_bproc.replace_materials(text_mat)
            
            text_distractors.append(text_bproc)

    if ENABLE_DETERMINISTIC:
        all_specific_meshes = []
        for c_name, m_list in valid_classes.items():
            for m in m_list:
                all_specific_meshes.append((c_name, m))
        random.shuffle(all_specific_meshes)
        mesh_dealer = itertools.cycle(all_specific_meshes)

    print(f"Starting Domain Randomization for {NUM_IMAGES} images...")

    for i in range(NUM_IMAGES):
        
        if i % 100 == 0:
            total, used, free = shutil.disk_usage(BASE_PATH)
            free_gb = free / (1024 ** 3) 
            if free_gb < 100.0:
                print(f"\n[ALERT] Only {free_gb:.2f} GB free space remaining! Stopping safely.")
                break 

        rand_bg_type = random.random()
        
        if rand_bg_type < 0.15:
            checker_mat = bproc.material.create(f"WeirdChecker_{i}")
            nodes = checker_mat.blender_obj.node_tree.nodes
            links = checker_mat.blender_obj.node_tree.links
            bsdf = nodes.get("Principled BSDF")
            checker = nodes.new('ShaderNodeTexChecker')
            
            checker.inputs['Color1'].default_value = [random.random(), random.random(), random.random(), 1]
            checker.inputs['Color2'].default_value = [random.random(), random.random(), random.random(), 1]
            checker.inputs['Scale'].default_value = random.uniform(2, 20)
            
            links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])
            bg_plane.replace_materials(checker_mat)

        elif rand_bg_type < 0.4:
            bg_mat = bproc.material.create_material_from_texture(random.choice(bg_images), f"bg_material_{i}")
            color = [random.uniform(0.1, 0.5), random.uniform(0.1, 0.5), random.uniform(0.1, 0.8), 1]
            bg_mat.set_principled_shader_value("Base Color", color)
            bg_plane.replace_materials(bg_mat)
            
        else:
            random_bg_path = random.choice(bg_images)
            bg_mat = bproc.material.create_material_from_texture(random_bg_path, f"bg_material_emissive_{i}")
            bg_mat.make_emissive(emission_strength=random.uniform(0.5, 2.0), emission_color=[1, 1, 1, 1]) 
            bg_plane.replace_materials(bg_mat)

        if random.random() < MULTI_CLASS_CHANCE:
            num_meshes_to_spawn = random.randint(2, MULTI_CLASS_MAX)
        else:
            num_meshes_to_spawn = 1
        
        active_meshes = []
        frame_classes = []
        placed_locations = []
        
        for _ in range(num_meshes_to_spawn):
            if ENABLE_DETERMINISTIC:
                chosen_class, chosen_mesh = next(mesh_dealer)
            else:
                chosen_class = random.choice(list(valid_classes.keys()))
                chosen_mesh = random.choice(valid_classes[chosen_class])
            
            chosen_mesh.set_rotation_euler(bproc.sampler.uniformSO3())
            
            target_loc = np.array([0.0, 0.0, 0.0]) 
            for attempt in range(50): 
                candidate_loc = np.random.uniform([-0.5, -0.5, -0.2], [0.5, 0.5, 0.2])
                
                if not placed_locations:
                    target_loc = candidate_loc
                    break
                    
                distances = [np.linalg.norm(candidate_loc - loc) for loc in placed_locations]
                if min(distances) >= MIN_MESH_DISTANCE:
                    target_loc = candidate_loc
                    break
                else:
                    target_loc = candidate_loc 
            
            placed_locations.append(target_loc)
            
            if ENABLE_MOTION_BLUR:
                chosen_mesh.set_location(target_loc, frame=0)
                drift = np.random.uniform([-MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT], 
                                          [MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT])
                chosen_mesh.set_location(target_loc - drift, frame=-1)
                bpy.context.scene.render.motion_blur_shutter = np.random.uniform(0.1, 0.6)
            else:
                chosen_mesh.set_location(target_loc)
                
            scale = np.random.uniform(TARGET_SCALE_RANGE[0], TARGET_SCALE_RANGE[1]) 
            chosen_mesh.set_scale([scale, scale, scale])
            
            if ENABLE_PBR_RANDOMIZATION and pbr_materials:
                is_animal = "animal" in chosen_class.lower()
                if is_animal and organic_pbr:
                    random_pbr = random.choice(organic_pbr)
                elif not is_animal and artificial_pbr:
                    random_pbr = random.choice(artificial_pbr)
                else:
                    random_pbr = random.choice(pbr_materials)
                chosen_mesh.replace_materials(random_pbr)
                
            elif ENABLE_HSV_HUE_SHIFT:
                for mat_slot in chosen_mesh.blender_obj.material_slots:
                    mat = mat_slot.material
                    if mat and mat.node_tree:
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        bsdf = nodes.get("Principled BSDF")
                        if bsdf:
                            hsv_node = nodes.get("DynamicHSV")
                            if not hsv_node:
                                base_color_input = bsdf.inputs.get("Base Color")
                                if base_color_input and base_color_input.links:
                                    original_link = base_color_input.links[0]
                                    from_socket = original_link.from_socket
                                    links.remove(original_link)
                                    
                                    hsv_node = nodes.new('ShaderNodeHueSaturation')
                                    hsv_node.name = "DynamicHSV"
                                    
                                    links.new(from_socket, hsv_node.inputs['Color'])
                                    links.new(hsv_node.outputs['Color'], base_color_input)
                            
                            if hsv_node:
                                hsv_node.inputs['Hue'].default_value = random.uniform(0.0, 1.0)
                                hsv_node.inputs['Saturation'].default_value = random.uniform(0.5, 1.5)
                                hsv_node.inputs['Value'].default_value = random.uniform(0.5, 1.5)

            elif ENABLE_TEXTURE_RANDOMIZATION:
                mats = chosen_mesh.get_materials()
                for mat in mats:
                    mat.set_principled_shader_value("Base Color", [random.random(), random.random(), random.random(), 1.0])
                    mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
                    mat.set_principled_shader_value("Metallic", np.random.uniform(0.0, 1.0))
                
            active_meshes.append(chosen_mesh)
            frame_classes.append(chosen_class)

        for distractor in distractors:
            target_loc = np.random.uniform([-1.2, -1.2, -0.5], [1.2, 1.2, 1.0])
            distractor.set_rotation_euler(bproc.sampler.uniformSO3())
            
            if ENABLE_MOTION_BLUR:
                distractor.set_location(target_loc, frame=0)
                drift_d = np.random.uniform([-MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT], 
                                            [MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT])
                distractor.set_location(target_loc - drift_d, frame=-1)
            else:
                distractor.set_location(target_loc)
                
            scale = np.random.uniform(0.05, 0.4)
            distractor.set_scale([scale, scale, scale])
            
            if ENABLE_PBR_RANDOMIZATION and pbr_materials:
                distractor.replace_materials(random.choice(pbr_materials))
            else:
                dist_mats = distractor.get_materials()
                if dist_mats:
                    dist_mats[0].set_principled_shader_value("Base Color", [random.random(), random.random(), random.random(), 1])
                    dist_mats[0].set_principled_shader_value("Metallic", random.choice([0.0, 1.0]))

        if ENABLE_TEXT_DISTRACTORS:
            for text_obj in text_distractors:
                if random.random() < 0.3:
                    b_loc = [100, 100, 100]
                    if ENABLE_MOTION_BLUR:
                        text_obj.set_location(b_loc, frame=0)
                        text_obj.set_location(b_loc, frame=-1)
                    else:
                        text_obj.set_location(b_loc)
                    continue
                    
                target_loc = np.random.uniform([-1.5, -1.5, -0.2], [1.5, 1.5, 1.5])
                text_obj.set_rotation_euler(bproc.sampler.uniformSO3())
                
                if ENABLE_MOTION_BLUR:
                    text_obj.set_location(target_loc, frame=0)
                    drift_t = np.random.uniform([-MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT, -MOTION_BLUR_AMOUNT], 
                                                [MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT, MOTION_BLUR_AMOUNT])
                    text_obj.set_location(target_loc - drift_t, frame=-1)
                else:
                    text_obj.set_location(target_loc)
                text_obj.set_scale([np.random.uniform(0.08, 0.3)] * 3)

        for light in lights:
            light.set_location(np.random.uniform([-3, -3, 1], [3, 3, 5]))
            light.set_energy(np.random.uniform(50, 1500)) 
            
            if LIGHT_COLOR_MODE == "Deep_Blue_Green":
                r = random.uniform(0.0, 0.2)
                g = random.uniform(0.3, 0.8)
                b = random.uniform(0.4, 1.0)
            else:
                r = random.uniform(0.1, 1.0)
                g = random.uniform(0.1, 1.0)
                b = random.uniform(0.1, 1.0)
            light.set_color([r, g, b])
            
            light.blender_obj.data.use_shadow = ENABLE_SHADOWS

        base_poi = np.array(bproc.object.compute_poi(active_meshes)) if active_meshes else np.array([0, 0, 0])
        
        if ENABLE_EXTREME_OFFSETS:
            if random.random() < NEGATIVE_SAMPLE_RATE: 
                poi_offset = np.random.uniform([-1.8, -1.8, -0.5], [1.8, 1.8, 0.5])
            else: 
                poi_offset = np.random.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2])
        else:
            poi_offset = np.array([0.0, 0.0, 0.0])
            
        location = bproc.sampler.shell(
            center=[0, 0, 0], 
            radius_min=CAMERA_RADIUS_RANGE[0], radius_max=CAMERA_RADIUS_RANGE[1],
            elevation_min=CAMERA_ELEVATION_RANGE[0], elevation_max=CAMERA_ELEVATION_RANGE[1] 
        )
        
        rotation_matrix = bproc.camera.rotation_from_forward_vec((base_poi + poi_offset) - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        bproc.camera.add_camera_pose(cam2world_matrix, frame=0)

        cam_data = bpy.context.scene.camera.data
        cam_data.dof.use_dof = True
        distance_to_target = np.linalg.norm((base_poi + poi_offset) - location)
        cam_data.dof.focus_distance = distance_to_target + np.random.uniform(-0.5, 0.5)
        cam_data.dof.aperture_fstop = np.random.uniform(1.0, 15.0)

        data = bproc.renderer.render()
        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

        chunk_id = i // 1000
        current_chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id:03d}")
        
        bproc.writer.write_coco_annotations(
            os.path.join(current_chunk_dir, 'coco_data'),
            instance_segmaps=seg_data["instance_segmaps"],
            instance_attribute_maps=seg_data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="JPEG",
            append_to_existing_output=True 
        )
        
        bproc.utility.reset_keyframes()

        json_path = os.path.join(current_chunk_dir, 'coco_data', 'coco_annotations.json')
        final_print_name = "background_image.jpg"
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                coco = json.load(f)
            
            if coco.get('images'):
                last_img = coco['images'][-1]
                old_file_name = last_img['file_name'] 
                
                rel_dir, base_file = os.path.split(old_file_name)
                prefix = "_".join(frame_classes) if frame_classes else "background"
                new_base_file = f"{prefix}_{base_file}"
                new_rel_path = os.path.join(rel_dir, new_base_file).replace('\\', '/') 
                
                old_abs_path = os.path.join(current_chunk_dir, 'coco_data', old_file_name)
                new_abs_path = os.path.join(current_chunk_dir, 'coco_data', rel_dir, new_base_file)
                
                if os.path.exists(old_abs_path):
                    os.rename(old_abs_path, new_abs_path)
                    last_img['file_name'] = new_rel_path
                    
                    with open(json_path, 'w') as f:
                        json.dump(coco, f)
                        
                    final_print_name = new_base_file

        for obj in active_meshes:
            obj.set_location([1000, 1000, 1000])
        
        print(f"Generated image {i+1}/{NUM_IMAGES} - Saved as {final_print_name}")

    if os.path.exists(temp_gltf):
        os.remove(temp_gltf)

    print(f"\nGeneration run finished! dataset safely stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()