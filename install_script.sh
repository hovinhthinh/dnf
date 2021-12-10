pip install pytorch

# from hoverboard
# conda activate pytorch_latest_p37

pip install matplotlib
pip install --ignore-installed llvmlite==0.36.0
pip install umap-learn==0.5.1
pip install --ignore-installed packaging==21.0
pip install datasets==1.11.0
pip install transformers==4.9.2
pip install tokenizers==0.10.3
# downgrade numpy to support
pip install numba==0.53.1
pip install numpy==1.19


# copy faiss from hoverdesktop
conda install -c pytorch faiss-gpu
