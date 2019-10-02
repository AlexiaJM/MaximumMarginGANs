#!/bin/bash

## Put datasets in node so it will be faster than reading from network.
## Assuming /project/rpp-bengioy/jolicoea/ -> /scratch/jolicoea/Datasets/
## Example:
# bash startup.sh  dir1="CIFAR10" dir2="Meow_64x64" dir3="Meow_128x128" dir4=""

# Arg1: Input_folder1
# Arg2: Input_folder2
# Arg3: Input_folder3
# Arg4: Input_folder4

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            dir1)    dir1=${VALUE} ;;
            dir2)    dir2=${VALUE} ;;
            dir3)    dir3=${VALUE} ;;
            dir4)    dir4=${VALUE} ;;
            *)   
    esac    

done

echo "Setting up fid_stats"
mkdir -p "$SLURM_TMPDIR/Datasets"
cp -r -n "/scratch/jolicoea/fid_stats" "$SLURM_TMPDIR"

if [ -z "$dir1" ];
then
	echo "Empty directory 1"
 
else
	echo "Setting up directory $dir1"
    mkdir -p "$SLURM_TMPDIR/Datasets/$dir1" && tar xzf "/project/rpp-bengioy/jolicoea/Datasets/$dir1.tar.gz" -C "$SLURM_TMPDIR/Datasets"
fi
if [ -z "$dir2" ];
then
	echo "Empty directory 2"
 
else
	echo "Setting up directory $dir2"
    mkdir -p "$SLURM_TMPDIR/Datasets/$dir2" && tar xzf "/project/rpp-bengioy/jolicoea/Datasets/$dir2.tar.gz" -C "$SLURM_TMPDIR/Datasets"
fi
if [ -z "$dir3" ];
then
	echo "Empty directory 3"
 
else
	echo "Setting up directory $dir3"
    mkdir -p "$SLURM_TMPDIR/Datasets/$dir3" && tar xzf "/project/rpp-bengioy/jolicoea/Datasets/$dir3.tar.gz" -C "$SLURM_TMPDIR/Datasets"
fi
if [ -z "$dir4" ];
then
	echo "Empty directory 4"
 
else
	echo "Setting up directory $dir4"
    mkdir -p "$SLURM_TMPDIR/Datasets/$dir4" && tar xzf "/project/rpp-bengioy/jolicoea/Datasets/$dir4.tar.gz" -C "$SLURM_TMPDIR/Datasets"
fi

# Make local directories in tempdir
mkdir -p $SLURM_TMPDIR/Output/Extra/

# Transfer Inception model
mkdir -p $SLURM_TMPDIR/models/
cp -r -n /scratch/jolicoea/models/Inception $SLURM_TMPDIR/models/

# Export directory
export SLURM_TMPDIR