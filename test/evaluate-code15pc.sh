# Download model (if you have not done it already)
curl https://zenodo.org/record/7038219/files/af_pred_model.zip?download=1 --output model.zip
unzip model.zip # the folder containing the model will be named model_fdset_30Mar

# Download CODE-15% part i=0
i=0
curl https://zenodo.org/record/4916206/files/exams_part"$i".zip?download=1 --output exams_part"$i".zip

# Evaluate model on CODE 15pc part i=0
python evaluate.py model_fdset_30Mar code15pc/exams_part0.hdf5 --traces_dset tracings --ids_dset exam_id --output exams_part0.csv

# Check the results are the same
python test/compare_results.py exams_part0.csv results/results-on-code.csv