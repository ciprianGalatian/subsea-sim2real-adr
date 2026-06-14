Hey there! Welcome to the Domain Randomization and Sim-to-Real Detection Pipeline.

This setup uses BlenderProc to generate synthetic data and YOLOv8/RT-DETR to train models on it. The main goal here is to bridge the sim-to-real gap by throwing heavy domain randomization at the problem.

Before you run anything, you'll need to set up your data folder. We keep all the data out of version control to keep the repository light, so just make sure you structure your 'data' folder exactly like this before starting:

1. 3D meshes
Create a 'meshes' folder inside 'data'. Inside that, make a new folder for every class you want to detect (like 'bottle' or 'tire'). Just drop your .glb 3D files into their matching class folders. Whatever you name these folders will become the official class names.

2. Background images
Create a 'backgrounds' folder inside 'data'. Throw in whatever .jpg or .png images you want to use as random 2D backgrounds. If you're doing marine debris, cropped underwater photos work perfectly here.

3. Real target dataset
Create a 'pure_yolo_dataset' folder inside 'data'. This is your real-world test data in standard YOLO format (with 'images' and 'labels' folders split into 'train' and 'val'). The most important thing here is to include a 'classes.txt' file right in this root folder. The names in that text file have to perfectly match the class folders you made in the meshes step.

4. PBR materials (Optional)
If you're planning to use the 'pbr' texture mode when generating data, you'll need to grab the materials first. Just open your terminal in the root folder and run:
blenderproc download cc_textures data/cc_materials

How to run

Step 1: Generate the synthetic data
First, let's generate the synthetic data. Run the generator script to create your randomized dataset. You can change things like the number of images or texture styles using command line flags. For example, you can run:

python src/generation/generator.py --num_images 5000 --texture_mode neon

That will spit out a timestamped folder inside your data directory with all the chunked, auto-labeled COCO datasets.

Step 2: Train
Next, it's time to train. Just run:

python src/training/train.py

The script will spot any synthetic datasets you've generated in the data folder and ask you which one you want to use. It automatically maps your synthetic classes to your golden real dataset and sets up the 80/20 train/val split for you.

Once it's done, you'll find all your outputs, model weights, and evaluation metrics inside the data/training_runs/ folder.
