# Create a directory to store the dataset
mkdir coco2017 && cd coco2017

# Download train and validation images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip train and validation images
unzip train2017.zip
unzip val2017.zip

# Unzip annotations
unzip annotations_trainval2017.zip

# Remove zip files to free up space
rm *.zip
