STG-OT：Spatiotemporal Graph neural networks (GNNs) Ordinary Differential Equations (ODEs) with unbalanced optimal transport
1.The model that learns continuous tissue dynamics by coupling cellular neighborhood graphs with neural differential equations and unbalanced optimal transport, directly addressing this critical gap

2.System Requirements and Software Dependencies
Python==3.8.20
All Python package dependencies are listed in requirements.txt.
No non-standard hardware is required. 

3.Installation
Clone the repository (or download the ZIP file and extract it):
  git clone https://github.com/Qiaorui_code/STG-OT.git
  cd STG-OT
or Install all required dependencies: requirements.txt
4.Getting started
STORIES takes as an input an AnnData object, where omics information and spatial coordinates are stored in obsm, 
and obs contains time information.
4.1 Data Prepossessing
All the code in the GAE.
4.2 Expression Predicted model
run main.py
4.2 spatial coordinates model
run cnf_OT.py
4.3 cell growth
run growth.py
