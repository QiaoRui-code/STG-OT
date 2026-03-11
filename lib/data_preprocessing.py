import scvelo as scv
import scanpy as sc
import matplotlib
matplotlib.use('Agg')

adata = sc.read_h5ad('/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/data/erythroid_lineage.h5ad')

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=1000)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
scv.tl.velocity(adata, mode="stochastic")
scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(adata, basis="umap", color="celltype")

scv.pl.velocity_embedding_stream(adata, basis="pca", color="celltype")

adata.obs["sample_labels"] = adata.obs["celltype"].replace(
    {
        "Blood progenitors 1": 0,
        "Blood progenitors 2": 1,
        "Erythroid1": 2,
        "Erythroid2": 3,
        "Erythroid3": 4,
    }
)

adata.write_h5ad("/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/results/adata.h5ad")