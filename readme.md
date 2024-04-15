# HTGN: Hetero Temporal Graph Neural Network

This project seeks to implement a heterogeneous version of the TGN model. In addition to handling heterogeneity, the GRU cell of the original TGN is replaced with Mamba and the graph attention convolution used by TGN has been upgraded to use Flash Attention.
This should produce a model architecture capable of the following:
* Handle heterogeneous data models like those found in real world data warehouses with full expression of node/edge create, update, or delete transactions
* Track much longer temporal patterns of the graph thanks to the Mamba architecture
* Faster calculation of Transformer GNN while still being able to utilize attention-driven explanations of GNN results

Experiments on replacing the Transformer GNN with a Co-operative GNN are also being considered

## Design
While the TGN paper passed messages through an RNN before a GNN, it did not address how to handle batches where a node 
was the destination of more than one message in a batch. This is critical to address, primarily in the case of node 
creation. When a new node appears, in many use cases, the node already has multiple edges and they all will be created 
within the same time step. This model architecture handles this by using a heterogeneous transformer convolution to 
aggregate messages into the destination node. Think of this as one round of partial graph message passing within a batch.

`Batch Message Passing -> RNN -> GNN`

### Model Process
1. `Data Loader` loads batches of new messages and creates additional messages for supporting maintenance, such as 
creating edge deletes for a node when the node is deleted. 
2. `Graph Store` holds all edges that have been loaded so far and is used to calculate `degree` or any other additional 
features.
3. `Feature Store` holds the current state of node features. Messages are enriched with the current features of a node 
to provide full context.
4.  Heterogeneous messages are encoded into a homogenous embedding space via `MessageEncoder`.
5.  Messages are aggregated for the batch by destination node ID via `Node Aggregator`.
6.  `Mamba Block` uses the aggregated messages and the destination node's previous state to calculate a new node 
"memory" per node and stores the memory in a `Vector Store`.
7.  `Transformer GNN` performs link prediction by using the edges in the `Graph Store` and the current node memories in 
the `Vector Store`.

### Message Format
All messages are communicated as edges. Self-Loop edges are used for node creation, updates, or deletions. 
* Time
* Action - CREATE, UPDATE, or DELETE
* Entity - NODE or EDGE
* Source Node Type
* Source Features
* Edge Type
* Edge Features
* Destination Node Type
* Destination Features


### Notes
* **Directed vs Undirected**: HTGN is inherently directional. For undirected graphs, messages can be duplicated with directions reversed
* **Update Transactions** - Self-Loop edges are used for node creation, updates, or deletions
* **Soft vs Hard Deletes** - Some graphs may benefit from processing delete messages but not actually removing the nodes/edges from the graph. For example, if a malicious user account was deleted, it may still be beneficial to preserve the user's relationships and data in the graph. Because of this, soft deletes are provided as an option. With hard deletes, the node/edge data will not be consumed at inference time. With soft deletes, delete messages are still processed and available at inference time. It still expected that a deleted edge/node will not see updates in the future.