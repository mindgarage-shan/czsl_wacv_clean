# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CURRENT_DIR=$(pwd)

FOLDER=$1

mkdir $FOLDER
# mkdir $FOLDER/fast
# mkdir $FOLDER/glove
# mkdir $FOLDER/w2v

# cp utils/download_embeddings.py $FOLDER/fast


# Download everything

wget --show-progress -O $FOLDER/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip

echo "Data downloaded. Extracting files..."
sed -i "s|ROOT_FOLDER|$FOLDER|g" utils/reorganize_utzap.py
sed -i "s|ROOT_FOLDER|$FOLDER|g" flags.py

# Dataset metadata, pretrained SVMs and features, tensor completion data


# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

# C-GQA
unzip cgqa.zip -d cgqa/

# Download new splits for Purushwalkam et. al
tar -zxvf splits.tar.gz

# remove all zip files and temporary files
rm -r attr-ops-data.tar.gz mitstates.zip utzap.zip splits.tar.gz cgqa.zip

# Download embeddings

# Glove (from attribute as operators)
mv data/glove/* glove/

# FastText
cd fast
python download_embeddings.py
rm cc.en.300.bin.gz

# Word2Vec
cd ../w2v
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
rm GoogleNews-vectors-negative300.bin.gz

cd ..
rm -r data

cd $CURRENT_DIR
python utils/reorganize_utzap.py
