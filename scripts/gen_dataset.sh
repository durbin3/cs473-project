if [ ! -d "dataset/images/resized_images" ]
then
    echo "Resizing Images" && python src/image_resizing.py;
fi

if [ -d "dataset/images/labeled_images" ]
then
    echo "Removing Existing Combined Folder"
    rm -rf dataset/images/labeled_images
fi

if [ -d "dataset/train" ]
then
    echo "Removing Existing train/test datasets"
    rm -rf dataset/train
    rm -rf dataset/test
fi

if [ -d "dataset/train.record" ]
then
    rm dataset/train.record
    rm dataset/test.record
fi

mkdir -p dataset/images/labeled_images
for file in `ls ./dataset/labels`;
do
    cp dataset/labels/$file dataset/images/labeled_images/$file
    image=$(find dataset/images/resized_images -name ${file:0:3}*)
    cp $image dataset/images/labeled_images/${image: -7}
done
python src/od_partition_dataset.py -i dataset/images/labeled_images/ -o dataset -r .3 -x
python src/generate_tfrecord.py -x dataset/train -l dataset/label_map.pbtxt -o dataset/train.record
python src/generate_tfrecord.py -x dataset/test -l dataset/label_map.pbtxt -o dataset/test.record