library(readr)
library(tidyverse)

mat <- read_tsv('Raw genetic interaction datasets_ Matrix format/Data File S2. Raw genetic interaction datasets: Matrix format/SGA_NxN_clustered.cdt')

amat <- mat %>%
  # extract data rows and columns
  slice(-(1:5)) %>%
  select(starts_with("dma")) %>%
  
  # convert to numeric matrix
  type_convert() %>%
  as.matrix()

# turn into unweighted, undirected adjacency matrix  
amat <- 1 * (abs(amat) > 0.2)


coldata <- mat %>%
  slice(1:5) %>%
  select(GID, starts_with("dma")) %>%
  slice(2) %>% select(-1) %>%
  gather(dma, orf, starts_with("dma"))

# extract ORF ids for queries (rows)
row_orf <- mat %>%
  slice(-(1:5)) %>%
  pull(ORF)

# match row and column ORF ids
m <- match(row_orf, coldata$orf)
rows_to_use <- !is.na(m)
cols_to_use <- m[rows_to_use]


orfs.repeat <- row_orf[rows_to_use][which(duplicated(row_orf[rows_to_use]))]
# subset matrix into ORFs found in both rows and columns
amat <- amat[rows_to_use, cols_to_use]

# set diagonal and missing entries in matrix to 0
diag(amat) <- 0
amat[is.na(amat)] <- 0

# make the adjacency matrix diagonal
amat <- ceiling(0.5 * (amat + t(amat)))
sum(abs(amat - t(amat))) == 0

######Question 1############
##Number of vertices
n.vertices = nrow(amat)

##Number of edges
n.edges = sum(amat)/2

##average degree
ave.degree = sum(amat)/n.vertices

##density
density = 2*n.edges/(n.vertices*(n.vertices-1))

#####Question 2##########
degrees = rowSums(amat)
###Frequency
h <- hist(degrees, breaks = 400, freq = T)

##prob
h <- hist(degrees, breaks = 400, freq = F)

plot(x = h$breaks[-1], y = h$density, log = 'xy')

####Question 3##########
library(dequer)

bfs.dist <- function(adj.mat, node)
{
  Q <- queue()
  pushback(Q, node)
  dist <- rep(Inf, ncol(adj.mat))
  distance <- 0
  dist[node] <- 0
  while(length(Q) != 0)
  {
    par <- pop(Q)
    neighbors <- which(adj.mat[par, ] == 1)
    distance = distance + 1
    for(n in neighbors)
    {
      
      if(dist[n] == Inf)
      {
        dist[n] = dist[par]+1
        pushback(Q, n)
      }
    }
  }
  return(dist)
}

bfs.dist(amat, 1)
library(parallel)
library(igraph)
graph <- graph_from_adjacency_matrix(amat)
distance.mat <- matrix(unlist(mclapply(seq(n.vertices), function(x) bfs.dist(amat,x), mc.cores = 2)), 
                       nrow = n.vertices, ncol = n.vertices, byrow = T)
distance.mat <- distances(graph, v = V(graph), mode = 'all')
max(distance.mat[distance.mat < Inf])
https://mathinsight.org/definition/network_mean_path_length