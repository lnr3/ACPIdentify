This is the user guide of running anticancer peptides prediction model for the nine specified categories.

First of all, download the Python source code, dataset and pretrained model to a Linux computer from https://github.com

Next, execute following commands:

unzip ACPIdentify.zip

cd ACPIdentify

pip install -r pip requirements.txt

cp esm_and_pdb.tar.gz GAT/

cp esm_and_pdb.tar.gz LightGBM/

cd GAT

tar -xzvf esm_and_pdb.tar.gz

cd ../LightGBM/

tar -xzvf esm_and_pdb.tar.gz

python TestLGBM.py

And now the program should be running and will output the prediction result for the test dataset on LightGBM model.
