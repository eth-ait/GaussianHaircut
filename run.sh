export GPU="0"
export CAMERA="PINHOLE"
export EXP_NAME_1="stage1"
export EXP_NAME_2="stage2"
export EXP_NAME_3="stage3"

#
# Ensure that the following environment variables are accessible to the script:
# PROJECT_DIR and DATA_PATH 
#

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

#################
# PREPROCESSING #
#################

# Arrange raw images into a 3D Gaussian Splatting format
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python preprocess_raw_images.py \
    --data_path $DATA_PATH

# Run COLMAP reconstruction and undistort the images and cameras
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python convert.py -s $DATA_PATH \
    --camera $CAMERA --max_size 1024

# Run Matte-Anything
conda deactivate && conda activate matte_anything
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
    --data_path $DATA_PATH --image_format png --max_size 2048

# Filter images using their IQA scores
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python filter_extra_images.py \
    --data_path $DATA_PATH --max_imgs 128

# Resize images
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python resize_images.py --data_path $DATA_PATH

# Calculate orientation maps
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
    --img_path $DATA_PATH/images_2 \
    --mask_path $DATA_PATH/masks_2/hair \
    --orient_dir $DATA_PATH/orientations_2/angles \
    --conf_dir $DATA_PATH/orientations_2/vars \
    --filtered_img_dir $DATA_PATH/orientations_2/filtered_imgs \
    --vis_img_dir $DATA_PATH/orientations_2/vis_imgs

# Run OpenPose
conda deactivate && cd $PROJECT_DIR/ext/openpose
mkdir $DATA_PATH/openpose
CUDA_VISIBLE_DEVICES="$GPU" ./build/examples/openpose/openpose.bin \
    --image_dir $DATA_PATH/images_4 \
    --scale_number 4 --scale_gap 0.25 --face --hand --display 0 \
    --write_json $DATA_PATH/openpose/json \
    --write_images $DATA_PATH/openpose/images --write_images_format jpg

# Run Face-Alignment
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_face_alignment.py \
    --data_path $DATA_PATH --image_dir "images_4"

# Run PIXIE
conda deactivate && conda activate pixie-env
cd $PROJECT_DIR/ext/PIXIE
CUDA_VISIBLE_DEVICES="$GPU" python demos/demo_fit_face.py \
    -i $DATA_PATH/images_4 -s $DATA_PATH/pixie \
    --saveParam True --lightTex False --useTex False \
    --rasterizer_type pytorch3d

# Merge all PIXIE predictions in a single file
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python merge_smplx_predictions.py \
    --data_path $DATA_PATH

# Convert COLMAP cameras to txt
conda deactivate && conda activate gaussian_splatting_hair
mkdir $DATA_PATH/sparse_txt
CUDA_VISIBLE_DEVICES="$GPU" colmap model_converter \
    --input_path $DATA_PATH/sparse/0  \
    --output_path $DATA_PATH/sparse_txt --output_type TXT

# Convert COLMAP cameras to H3DS format
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python colmap_parsing.py \
    --path_to_scene $DATA_PATH

# Remove raw files to preserve disk space
rm -rf $DATA_PATH/input $DATA_PATH/images $DATA_PATH/masks $DATA_PATH/iqa*

##################
# RECONSTRUCTION #
##################

export EXP_PATH_1=$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1

# Run 3D Gaussian Splatting reconstruction
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
    -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --lambda_dorient 0.1

# Run FLAME mesh fitting
conda activate gaussian_splatting_hair
cd $PROJECT_DIR/ext/NeuralHaircut/src/multiview_optimization

CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
    --batch_size 1 --train_rotation True --fixed_images True \
    --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1 \
    --data_path $DATA_PATH \
    --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
    --batch_size 4 --train_rotation True --fixed_images True \
    --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2 \
    --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1/opt_params_final \
    --data_path $DATA_PATH \
    --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1_.conf \
    --batch_size 32 --train_rotation True --train_shape True \
    --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3 \
    --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2/opt_params_final \
    --data_path $DATA_PATH \
    --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

# Crop the reconstructed scene
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python scale_scene_into_sphere.py \
    --path_to_data $DATA_PATH \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000

# Remove hair Gaussians that intersect with the FLAME head mesh
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python filter_flame_intersections.py \
    --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000 \
    --project_dir $PROJECT_DIR/ext/NeuralHaircut

# Run rendering for training views
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python render_gaussians.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" \
    --skip_test --scene_suffix "_cropped" --iteration 30000 \
    --trainable_cameras --trainable_intrinsics --use_barf

# Get FLAME mesh scalp maps
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python extract_non_visible_head_scalp.py \
    --project_dir $PROJECT_DIR/ext/NeuralHaircut --data_dir $DATA_PATH \
    --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
    --cams_path $DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/cameras/30000_matrices.pkl \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1"

# Run latent hair strands reconstruction
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
    --model_path_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2" \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --iterations 20000 --port "800$GPU"

# Run hair strands reconstruction
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_strands.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
    --model_path_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
    --start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
    --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --iterations 10000 --port "800$GPU"

rm -rf "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/train_cropped"

##################
# VISUALIZATIONS #
##################

# Export the resulting strands as pkl and ply
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python export_curves.py \
    --data_dir $DATA_PATH --model_name $EXP_NAME_3 --iter 10000 \
    --flame_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3/mesh_final.obj" \
    --scalp_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/scalp_data/scalp.obj" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml"

# Render the visualizations
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python render_video.py \
    --blender_path "$BLENDER_DIR" --input_path "$DATA_PATH" \
    --exp_name_1 "$EXP_NAME_1" --exp_name_3 "$EXP_NAME_3"

# Render the strands
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python render_strands.py \
    -s $DATA_PATH --data_dir "$DATA_PATH" --data_device 'cpu' --skip_test \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iteration 30000 \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --model_hair_path "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
    --checkpoint_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3/checkpoints/10000.pth" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud/iteration_30000/raw_point_cloud.ply" \
    --interpolate_cameras

# Make the video
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python concat_video.py \
    --input_path "$DATA_PATH" --exp_name_3 "$EXP_NAME_3"