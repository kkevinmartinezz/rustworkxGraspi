import rustworkx as rx
import math

from graphviz.dot import subgraph, graph_head
from numpy.ma.core import append
from rustworkx import dfs_search, dfs_edges, bfs_search, dijkstra_shortest_paths, PyGraph, PyDiGraph, dijkstra_search
from rustworkx.visualization import mpl_draw
import matplotlib.pyplot as plt
import graphviz
from rustworkx.visualization import graphviz_draw
import time
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

from rustworkx.visit import BFSVisitor, DijkstraVisitor
from rustworkx.visit import DFSVisitor


test = "tests/test.txt"
file10 = "tests/10x10.txt"
file50 = "tests/test50.txt"
file100 = "tests/test100.txt"
file500 = "tests/test500.txt"
file1000 = "tests/test1000.txt"

file3D10 = "tests/10x10x10.txt"
file3D50 = "tests/50x50x50.txt"
file3D100 = "tests/100x100x100.txt"

class TreeEdgesRecorderDfs(DFSVisitor):

    def __init__(self):
        self.edges = []

    def tree_edge(self, edge):
        self.edges.append(edge)

class TreeEdgesRecorderBfs(BFSVisitor):

    def __init__(self):
        self.edges = []

    def tree_edge(self, edge):
        self.edges.append(edge)

class Node:
  def __init__(self, label, color, x, y, z):
      self.label = label
      # Color of the node depending on 0 or 1
      self.color = color
      # Coordinates of the node
      self.x = x
      self.y = y
      self.z = z

class Edge:
    def __init__(self, node1, node2, weight):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

graph = rx.PyGraph()

def createGraph(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        header = lines[0].split()

        # Dimensions of the graph
        dimX, dimY, dimZ = map(int, header)

        # Stores necessary data for creating edges
        prevLayer = [[None] * dimX for i in range(dimY)]
        currLayer = [[None] * dimX for i in range(dimY)]
        prevRow = [None] * dimX  # Allocates necessary space beforehand
        currRow = [None] * dimX  # Allocates necessary space beforehand
        prevNode = None

        line_idx = 1
        cathode = graph.add_node(Node("Cathode",2,0,0,0))
        # Graph creation
        for z in range(dimZ):
            for y in range(dimY):
                line = lines[line_idx].strip().split(" ")
                line_idx += 1
                for x in range(dimX):
                    node = graph.add_node(Node((z * dimX * dimY) + (y * dimX) + x, int(line[x]), x, y, z))
                    currRow[x] = node
                    currLayer[y][x] = node

                    if y == 0:
                        graph.add_edge(node, cathode,Edge(node,cathode,1))
                    # Left of the node
                    if prevNode != None:
                        graph.add_edge(node, prevNode, Edge(node, prevNode, 1))

                    # Down of the node
                    if prevRow[x] != None:
                        graph.add_edge(node, prevRow[x], Edge(node, prevRow[x], 1))

                    if (prevLayer[y][x] != None):
                        graph.add_edge(node, prevLayer[y][x], Edge(node, prevLayer[y][x], 1))

                    # Southeast of the node
                    if (x + 1 < dimX) and (prevRow[x + 1] != None):
                        graph.add_edge(node, prevRow[x + 1], Edge(node, prevRow[x + 1], math.sqrt(2)))

                    # Southwest of the node
                    if (x - 1 >= 0) and (prevRow[x - 1] != None):
                        graph.add_edge(node, prevRow[x - 1], Edge(node, prevRow[x - 1], math.sqrt(2)))

                    # Checks if the node isn't the last node on the line
                    if (x < dimX - 1):
                        prevNode = node
                    else:
                        prevNode = None
                # Stores the previous row as the current row, clears current row
                prevRow, currRow = currRow, [None] * dimX

            prevLayer, currLayer = currLayer, [[None] * dimX for i in range(dimY)]
        # print(dimX)
        # print(dimY)
        # print(dimZ)
        #add_cathode_node(dimX, dimY, dimZ)

def add_cathode_node(dimX,dimY,dimZ):
    node = graph.add_node(Node())
    currNodes = graph.node_indices()
    for z in range(dimZ):
        for x in range(dimX):
            graph.add_edge(node, currNodes[x], Edge(node, currNodes[x], 1))


    return 0

def node_attr_fn(node):
    attr_dict = {
        "style": "filled",
        "shape": "circle",
        "label": str(node.label),
        "rank": "same"
    }
    # find out how to reach into Node class for color
    # if node is 0 make black, if 1 make white
    if node.color == 2:
        attr_dict["color"] = "blue"
        attr_dict["fillcolor"] = "blue"
        attr_dict["fontcolor"] = "white"
    elif node.color == 1:
        attr_dict["color"] = "black"
        attr_dict["fillcolor"] = "black"
        attr_dict["fontcolor"] = "white"
    else:
        attr_dict["color"] = "black"
        attr_dict["fillcolor"] = "white"
    return attr_dict

def visualizeGraphMPL(g):
    for node in graph.node_indices():
        graph[node] = node
    graphviz_draw(graph, node_attr_fn=node_attr_fn, method ="neato")

def visualizeGraphGV(g, file):
 # for node in graph.node_indices():
     # graph[node] = graph.get_node_data(node)
    graph_dict = {}
    graphviz_draw(g, filename=file, node_attr_fn=node_attr_fn,
                  graph_attr=graph_dict, method ="neato")


#createGraph(file1000)
# visualizeGraph()

def testGraphRunTime(filename, visualize, times):
    totalTime = 0
    if visualize:
        for i in range(times):
            start = time.time()
            createGraph(filename)
            visualizeGraphGV(graph, "images/rustworkx_graph.jpg")
            totalTime += time.time() - start
    else:
        for i in range(times):
            start = time.time()
            createGraph(filename)
            totalTime += time.time() - start
    print(totalTime / times)

def connectedComponents(edge):
    node1 = graph.get_node_data(edge.node1)
    node2 = graph.get_node_data(edge.node2)

    # Checks if the edge between the two nodes have different colors
    if ( (node1.color == 0 and node2.color == 1) or (node1.color == 1 and node2.color == 0) ):
        return False
    return True

filteredGraph = PyGraph()
def filterGraph(visualize):
    edges = graph.filter_edges(connectedComponents)
    edgeList = []

    for edge in edges:
        node1 = graph.get_edge_data_by_index(edge).node1
        node2 = graph.get_edge_data_by_index(edge).node2
        edgeList.append( (node1, node2) )
    #subgraph = graph.edge_subgraph(edgeList)
    filteredGraph = graph.edge_subgraph(edgeList)
    #filteredGraph.add_nodes_from(graph.nodes())
    if visualize:
        print(filteredGraph.edge_indices())
        visualizeGraphGV(filteredGraph, "images/rustworkx_subgraph.jpg")

    return edges

def testFilterGraph(filename, visualize, times):
    totalTime = 0
    for i in range(times):
        start = time.time()
        createGraph(filename)
        filterGraph(visualize)
        totalTime += time.time() - start
    print(totalTime / times)

#Uses DFS to traverse graph and print's all edges reachele from source node
def dfs(g, source):
    nodes = []
    nodes.append(source)
    visDFS = TreeEdgesRecorderDfs()
    rx.dfs_search(g, nodes, visDFS)
    print('DFS Edges:', visDFS.edges)

#Uses BFS to traverse graph and print's all edges reachele from source node
def bfs(g, source):
    nodes = []
    nodes.append(source)
    visBFS = TreeEdgesRecorderBfs()
    rx.bfs_search(g, nodes, visBFS)
    print('BFS Edges:', visBFS.edges)

#finds shortest path between a source and target node
def shortest_path(g, source, target):
    print('Shortest Path between', source, 'and', target , dijkstra_shortest_paths(g, source, target, weight_fn=None, default_weight=1))


# Defining main function
def main():
    testGraphRunTime(file10, True, 1)
    testFilterGraph(file10,True,1)
    print(graph.node_indices())
    bfs(graph, 2)
    dfs(graph,2)
    shortest_path(filteredGraph,0,0)

if __name__=="__main__":
    main()