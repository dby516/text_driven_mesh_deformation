###
### ShapeTalk provides you with three types of data:
###

#    1. Collected referential language ( 120 MB ) : 
#       https://shapetalk-public.s3.amazonaws.com/language.zip
#
#    2. Image-renderings used to constrast 3D objects ( 2.2 GB ):  
#       https://shapetalk-public.s3.amazonaws.com/images.zip
#
#    3. Extracted Point-Clouds from the Surface of the underlying 3D objects:
#       - scaled to respect the above rendering's relative sizes ( 10.9 GB ) : 
#         https://shapetalk-public.s3.amazonaws.com/point_clouds_scaled_to_align_rendering.zip


### Below is a bash script to download all three types mentioned above. 

### To "integrate" it to our shared CODEBASE: https://github.com/optas/changeit3d
### **MOVE this file** in the repo under changeit3d/changeit3d/scripts/bash_scripts/ and THEN run it

TOP_OUT_DIR="../../data/shapetalk"   # the default expected folder where the data should be placed for our codebase (above)
CURDIR="$PWD"

mkdir -p $TOP_OUT_DIR
cd $TOP_OUT_DIR
echo "**** Starting to download ShapeTalk at ${TOP_OUT_DIR} ****"

wget https://shapetalk-public.s3.amazonaws.com/images.zip
unzip images.zip
rm -rf images.zip

wget https://shapetalk-public.s3.amazonaws.com/language.zip
unzip language.zip
rm -rf language.zip

mkdir -p point_clouds
cd point_clouds
wget https://shapetalk-public.s3.amazonaws.com/point_clouds_scaled_to_align_rendering.zip
unzip point_clouds_scaled_to_align_rendering.zip
rm -rf point_clouds_scaled_to_align_rendering.zip
cd ..

cd $CURDIR
echo '**** Done ****'
