import networkx as nx
import os
import bolsas

numGrafos = 10
numNodosAle=40
probabilidad=0.4
path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'

'''Crea un grafo aleatorio'''
def crea_grafo_aleatorio(numNodos, prob,nombre):
    G = nx.fast_gnp_random_graph(numNodos,prob) #Genera un grafo aleatorio con un número de nodos
                                                    #y una cierta probabilidad
    nx.write_edgelist(G, nombre)


'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    #link=path + nombre
    return nx.read_edgelist(nombre)


for i in range(numGrafos):
    carpeta = path + '/Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    nombreGrafo='G' + str(numNodosAle) + '_' + str(probabilidad) + '_'
    nombreGrafo=nombreGrafo + str(i+1)
    carpeta= carpeta + "/" + nombreGrafo
    crea_grafo_aleatorio(numNodosAle,probabilidad,carpeta)
    G=Lee_Grafo(carpeta)
    print(G.edges)
    bolsas.Muestra_Grafo(G)