# TUM-RGBD Dataset
mkdir -p ./data/TUMRGBD
wget -P ./data/TUMRGBD/ -i ./data/rgbd_tum_datasets.txt -c

for filename in `ls data/TUMRGBD/ -l | awk '$1 {print $9}' `
do
echo data/TUMRGBD/$filename
tar -xzvf data/TUMRGBD/$filename -C data/TUMRGBD
done

for dirlist in `ls data/TUMRGBD/ -l | awk '$1 ~ /d/ {print $9}' `
do
echo $dirlist
python2 data/associate.py data/TUMRGBD/$dirlist/rgb.txt data/TUMRGBD/$dirlist/depth.txt > data/TUMRGBD/$dirlist/associations.txt
done

# ICL-NUIM Dataset
# mkdir -p ./data/ICLNUIM
