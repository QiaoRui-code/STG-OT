import scvelo as scv
import scanpy as sc
import matplotlib
matplotlib.use('Agg')

adata2 = sc.read_h5ad('/home/lenovo/jora/causual/subset_200_genes.h5ad')

scv.pp.filter_and_normalize(adata2, min_shared_counts=20, n_top_genes=200)
sc.pp.pca(adata2)
sc.pp.neighbors(adata2, n_pcs=30, n_neighbors=30)
scv.tl.velocity(adata2, mode="stochastic")
scv.tl.velocity_graph(adata2)

scv.pl.velocity_embedding_stream(adata2, basis="umap", color="celltype")

scv.pl.velocity_embedding_stream(adata2, basis="pca", color="celltype")


#adata2.write('Nadata.h5ad')

