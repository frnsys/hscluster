# HS Clustering
### A clique-based approach to clustering news articles

A write up is [available here](http://spaceandtim.es/projects/clustering_news_articles).

You can reproduce the results (to a degree - KMeans is stochastic) in the write-up by running:

    $ python compare.py

However, you will need to have the IDF, phrase, and Doc2Vec models (reach out if you're interested).

Here is a summary of the results:

    --------------------
    research/data/10E.json
    --------------------

    -------- dbscan_cluster
    Looking for 10 clusters
    Found 6 clusters
    Took 14.86 seconds
    Completeness 1.0
    Homogeneity 0.748593882768
    Adjusted Mutual Info 0.596475807189
    Adjusted Rand 0.4539506794162053

    -------- d2v_cluster
    Looking for 10 clusters
    Found 10 clusters
    Took 4.36 seconds
    Completeness 0.541345678522
    Homogeneity 0.514368694966
    Adjusted Mutual Info -0.00447890048626
    Adjusted Rand -0.0054054054054054005

    -------- kmeans_cluster
    Looking for 10 clusters
    Found 7 clusters
    Took 1.50 seconds
    Completeness 0.966021172957
    Homogeneity 0.761987331961
    Adjusted Mutual Info 0.607775542535
    Adjusted Rand 0.5275016567263089

    -------- hscluster
    Looking for 10 clusters
    Found 13 clusters
    Took 3.31 seconds
    Completeness 0.931912456987
    Homogeneity 1.0
    Adjusted Mutual Info 0.839738335718
    Adjusted Rand 0.9123203982350945


    --------------------
    research/data/20E.json
    --------------------

    -------- dbscan_cluster
    Looking for 20 clusters
    Found 16 clusters
    Took 0.50 seconds
    Completeness 0.958032101461
    Homogeneity 0.863789748619
    Adjusted Mutual Info 0.715324335082
    Adjusted Rand 0.5731624106880862

    -------- d2v_cluster
    Looking for 20 clusters
    Found 22 clusters
    Took 4.11 seconds
    Completeness 0.613478166476
    Homogeneity 0.575317816813
    Adjusted Mutual Info 0.0455666910698
    Adjusted Rand 0.02189961264737643

    -------- kmeans_cluster
    Looking for 20 clusters
    Found 15 clusters
    Took 6.02 seconds
    Completeness 0.942149662344
    Homogeneity 0.841711251806
    Adjusted Mutual Info 0.679291591101
    Adjusted Rand 0.6860993174075065

    -------- hscluster
    Looking for 20 clusters
    Found 35 clusters
    Took 3.94 seconds
    Completeness 0.871013416222
    Homogeneity 1.0
    Adjusted Mutual Info 0.63835561364
    Adjusted Rand 0.7457731153395458


    --------------------
    research/data/30E.json
    --------------------

    -------- dbscan_cluster
    Looking for 30 clusters
    Found 7 clusters
    Took 0.83 seconds
    Completeness 0.930692566979
    Homogeneity 0.256805359565
    Adjusted Mutual Info 0.121833043471
    Adjusted Rand 0.023120345363163063

    -------- hscluster
    Looking for 30 clusters
    Found 51 clusters
    Took 7.05 seconds
    Completeness 0.872221224627
    Homogeneity 0.976895887911
    Adjusted Mutual Info 0.604640648263
    Adjusted Rand 0.6878569192332628
