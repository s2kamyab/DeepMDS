# DeepMDS
The data and code for the paper "Deep-MDS Framework for Recovering the 3D Shape of 2D Landmarks from a Single Image".

In the case of BFM model, put the model from 
https://faces.dmi.unibas.ch/bfm/
in the folder "PublicMM1" and then you can generate the data by running the script "gen_data_BFM.m"

In the case of FLAME model, download Coma dataset from
https://coma.is.tue.mpg.de/
and put in the folder "CoMA_raw_data" and then run "gen_tr_tst_flame.py"
For data generation we used some code and data from 
https://github.com/TimoBolkart/TF_FLAME
and 
https://github.com/Rubikplayer/flame-fitting


In the case of Mediapipe library, download celebA aligned data and put in the folder "Mediapipe/data/celebA/img_align_celebA" and then run "save_lmks_from_mediapipe_output.py"
