## Requirements
# pip install opencv-python
# Install PyTorch from source 
## Optional
# Install Tensorflow from source (If you want to be able to calculate the FID)
# Install the latest version of R (If you want to recreate the plots of the bias like in the paper)

###### Important note: If downloading from archive doesn't work, download it from http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd (and then unzip the 6 zip files)
###### Try to still download https://archive.org/download/CAT_DATASET/00000003_015.jpg.cat, I don't know if it's corrected in the torrent

## Download CAT dataset from site
wget -nc https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip
wget -nc https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip
wget -nc https://archive.org/download/CAT_DATASET/00000003_015.jpg.cat

## Setting up folder ()
unzip CAT_DATASET_01.zip -d cat_dataset
unzip CAT_DATASET_02.zip -d cat_dataset
mv cat_dataset/CAT_00/* cat_dataset
rmdir cat_dataset/CAT_00
mv cat_dataset/CAT_01/* cat_dataset
rmdir cat_dataset/CAT_01
mv cat_dataset/CAT_02/* cat_dataset
rmdir cat_dataset/CAT_02
mv cat_dataset/CAT_03/* cat_dataset
rmdir cat_dataset/CAT_03
mv cat_dataset/CAT_04/* cat_dataset
rmdir cat_dataset/CAT_04
mv cat_dataset/CAT_05/* cat_dataset
rmdir cat_dataset/CAT_05
mv cat_dataset/CAT_06/* cat_dataset
rmdir cat_dataset/CAT_06

## Error correction
rm cat_dataset/00000003_019.jpg.cat
mv 00000003_015.jpg.cat cat_dataset/00000003_015.jpg.cat

## Removing outliers
# Corrupted, drawings, badly cropped, inverted, impossible to tell it's a cat, blocked face
cd cat_dataset
rm 00000004_007.jpg 00000007_002.jpg 00000045_028.jpg 00000050_014.jpg 00000056_013.jpg 00000059_002.jpg 00000108_005.jpg 00000122_023.jpg 00000126_005.jpg 00000132_018.jpg 00000142_024.jpg 00000142_029.jpg 00000143_003.jpg 00000145_021.jpg 00000166_021.jpg 00000169_021.jpg 00000186_002.jpg 00000202_022.jpg 00000208_023.jpg 00000210_003.jpg 00000229_005.jpg 00000236_025.jpg 00000249_016.jpg 00000254_013.jpg 00000260_019.jpg 00000261_029.jpg 00000265_029.jpg 00000271_020.jpg 00000282_026.jpg 00000316_004.jpg 00000352_014.jpg 00000400_026.jpg 00000406_006.jpg 00000431_024.jpg 00000443_027.jpg 00000502_015.jpg 00000504_012.jpg 00000510_019.jpg 00000514_016.jpg 00000514_008.jpg 00000515_021.jpg 00000519_015.jpg 00000522_016.jpg 00000523_021.jpg 00000529_005.jpg 00000556_022.jpg 00000574_011.jpg 00000581_018.jpg 00000582_011.jpg 00000588_016.jpg 00000588_019.jpg 00000590_006.jpg 00000592_018.jpg 00000593_027.jpg 00000617_013.jpg 00000618_016.jpg 00000619_025.jpg 00000622_019.jpg 00000622_021.jpg 00000630_007.jpg 00000645_016.jpg 00000656_017.jpg 00000659_000.jpg 00000660_022.jpg 00000660_029.jpg 00000661_016.jpg 00000663_005.jpg 00000672_027.jpg 00000673_027.jpg 00000675_023.jpg 00000692_006.jpg 00000800_017.jpg 00000805_004.jpg 00000807_020.jpg 00000823_010.jpg 00000824_010.jpg 00000836_008.jpg 00000843_021.jpg 00000850_025.jpg 00000862_017.jpg 00000864_007.jpg 00000865_015.jpg 00000870_007.jpg 00000877_014.jpg 00000882_013.jpg 00000887_028.jpg 00000893_022.jpg 00000907_013.jpg 00000921_029.jpg 00000929_022.jpg 00000934_006.jpg 00000960_021.jpg 00000976_004.jpg 00000987_000.jpg 00000993_009.jpg 00001006_014.jpg 00001008_013.jpg 00001012_019.jpg 00001014_005.jpg 00001020_017.jpg 00001039_008.jpg 00001039_023.jpg 00001048_029.jpg 00001057_003.jpg 00001068_005.jpg 00001113_015.jpg 00001140_007.jpg 00001157_029.jpg 00001158_000.jpg 00001167_007.jpg 00001184_007.jpg 00001188_019.jpg 00001204_027.jpg 00001205_022.jpg 00001219_005.jpg 00001243_010.jpg 00001261_005.jpg 00001270_028.jpg 00001274_006.jpg 00001293_015.jpg 00001312_021.jpg 00001365_026.jpg 00001372_006.jpg 00001379_018.jpg 00001388_024.jpg 00001389_026.jpg 00001418_028.jpg 00001425_012.jpg 00001431_001.jpg 00001456_018.jpg 00001458_003.jpg 00001468_019.jpg 00001475_009.jpg 00001487_020.jpg
rm 00000004_007.jpg.cat 00000007_002.jpg.cat 00000045_028.jpg.cat 00000050_014.jpg.cat 00000056_013.jpg.cat 00000059_002.jpg.cat 00000108_005.jpg.cat 00000122_023.jpg.cat 00000126_005.jpg.cat 00000132_018.jpg.cat 00000142_024.jpg.cat 00000142_029.jpg.cat 00000143_003.jpg.cat 00000145_021.jpg.cat 00000166_021.jpg.cat 00000169_021.jpg.cat 00000186_002.jpg.cat 00000202_022.jpg.cat 00000208_023.jpg.cat 00000210_003.jpg.cat 00000229_005.jpg.cat 00000236_025.jpg.cat 00000249_016.jpg.cat 00000254_013.jpg.cat 00000260_019.jpg.cat 00000261_029.jpg.cat 00000265_029.jpg.cat 00000271_020.jpg.cat 00000282_026.jpg.cat 00000316_004.jpg.cat 00000352_014.jpg.cat 00000400_026.jpg.cat 00000406_006.jpg.cat 00000431_024.jpg.cat 00000443_027.jpg.cat 00000502_015.jpg.cat 00000504_012.jpg.cat 00000510_019.jpg.cat 00000514_016.jpg.cat 00000514_008.jpg.cat 00000515_021.jpg.cat 00000519_015.jpg.cat 00000522_016.jpg.cat 00000523_021.jpg.cat 00000529_005.jpg.cat 00000556_022.jpg.cat 00000574_011.jpg.cat 00000581_018.jpg.cat 00000582_011.jpg.cat 00000588_016.jpg.cat 00000588_019.jpg.cat 00000590_006.jpg.cat 00000592_018.jpg.cat 00000593_027.jpg.cat 00000617_013.jpg.cat 00000618_016.jpg.cat 00000619_025.jpg.cat 00000622_019.jpg.cat 00000622_021.jpg.cat 00000630_007.jpg.cat 00000645_016.jpg.cat 00000656_017.jpg.cat 00000659_000.jpg.cat 00000660_022.jpg.cat 00000660_029.jpg.cat 00000661_016.jpg.cat 00000663_005.jpg.cat 00000672_027.jpg.cat 00000673_027.jpg.cat 00000675_023.jpg.cat 00000692_006.jpg.cat 00000800_017.jpg.cat 00000805_004.jpg.cat 00000807_020.jpg.cat 00000823_010.jpg.cat 00000824_010.jpg.cat 00000836_008.jpg.cat 00000843_021.jpg.cat 00000850_025.jpg.cat 00000862_017.jpg.cat 00000864_007.jpg.cat 00000865_015.jpg.cat 00000870_007.jpg.cat 00000877_014.jpg.cat 00000882_013.jpg.cat 00000887_028.jpg.cat 00000893_022.jpg.cat 00000907_013.jpg.cat 00000921_029.jpg.cat 00000929_022.jpg.cat 00000934_006.jpg.cat 00000960_021.jpg.cat 00000976_004.jpg.cat 00000987_000.jpg.cat 00000993_009.jpg.cat 00001006_014.jpg.cat 00001008_013.jpg.cat 00001012_019.jpg.cat 00001014_005.jpg.cat 00001020_017.jpg.cat 00001039_008.jpg.cat 00001039_023.jpg.cat 00001048_029.jpg.cat 00001057_003.jpg.cat 00001068_005.jpg.cat 00001113_015.jpg.cat 00001140_007.jpg.cat 00001157_029.jpg.cat 00001158_000.jpg.cat 00001167_007.jpg.cat 00001184_007.jpg.cat 00001188_019.jpg.cat 00001204_027.jpg.cat 00001205_022.jpg.cat 00001219_005.jpg.cat 00001243_010.jpg.cat 00001261_005.jpg.cat 00001270_028.jpg.cat 00001274_006.jpg.cat 00001293_015.jpg.cat 00001312_021.jpg.cat 00001365_026.jpg.cat 00001372_006.jpg.cat 00001379_018.jpg.cat 00001388_024.jpg.cat 00001389_026.jpg.cat 00001418_028.jpg.cat 00001425_012.jpg.cat 00001431_001.jpg.cat 00001456_018.jpg.cat 00001458_003.jpg.cat 00001468_019.jpg.cat 00001475_009.jpg.cat 00001487_020.jpg.cat
cd ..

## Preprocessing and putting in folders for different image sizes
mkdir cats_bigger_than_32x32
mkdir cats_bigger_than_64x64
mkdir cats_bigger_than_128x128
wget -nc https://raw.githubusercontent.com/AlexiaJM/Relativistic-f-divergences/master/preprocess_cat_dataset.py
python preprocess_cat_dataset.py

## Removing cat_dataset
rm -r cat_dataset

## Move to your favorite place
#mv cats_bigger_than_32x32 /home/alexia/Datasets/Meow_32x32
#mv cats_bigger_than_64x64 /home/alexia/Datasets/Meow_64x64
#mv cats_bigger_than_128x128 /home/alexia/Datasets/Meow_128x128

## Create FID stats
# Change to your folders

# CIFAR-10
# Note that I actually used the one in http://bioinf.jku.at/research/ttur/, but either way should be the same
python create_FID_stats.py --output_path '/home/alexia/fid_stats/CIFAR10_fid_stats32.npz'

# CAT 32x32
python create_FID_stats.py --data_path '/home/alexia/Datasets/Meow_32x32/cats_bigger_than_32x32' --output_path '/home/alexia/fid_stats/CAT_fid_stats32.npz'

# CelebA 32x32
python preprocess_dataset.py --centercrop 160 --image_size 32 --input_path '/home/alexia/Datasets/CelebA/img_align_celeba' --output_path '/home/alexia/Datasets/CelebA_transformed32'
python create_FID_stats.py --data_path '/home/alexia/Datasets/CelebA_transformed32' --output_path '/home/alexia/fid_stats/CelebA_fid_stats32.npz'