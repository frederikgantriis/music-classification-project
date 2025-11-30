# Fetch dataset from kaggle and save as a zip file
curl -L -o ./gtzan-dataset-music-genre-classification.zip https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification

# Unzip dataset
unzip gtzan-dataset-music-genre-classification.zip

# Move data folder into raw-data
mv Data raw-data

# Remove the original zip-file
rm gtzan-dataset-music-genre-classification.zip
