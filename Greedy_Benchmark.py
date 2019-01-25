import networkx as nx
import time

path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'
#Nombre_benchmark = ["myciel3","myciel4",'2-Fullins_3',"jean","anna","DSJC250-5","DSJC500-5","le450_15c","le450_25c",'flat300_28']
#k = [4,5,5,10,11,28,48,15,25,31]
k = [2,4,6,8,10,12]

#Datos del grafo aleatorio
numNodosAle = 100
probabilidad = 0.7
Estrategia='independent_set' # smallest_last independent_set



'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre)


'''Calcula el número monocromático del individuo'''
def numeroAristasMono(G, individuo, numColores,k):
    num_mono = 0
    for i in range(numColores): #Toma las bolsas del individuo
        if i == 48:
            print("ya!")
        nodos_bolsa = [k for k,v in individuo.items() if v == i]    #Toma los nodos contenidos en la bolsa actual
        for j in range(len(nodos_bolsa)-1): #Toma en orden los nodos
            r = j+1
            while r < len(nodos_bolsa): #Toma el nodo siguiente al actual
                if G.has_edge(nodos_bolsa[j], nodos_bolsa[r]) or G.has_edge(nodos_bolsa[r], nodos_bolsa[j]) or i > k-1: #Pregunta si hay una
                    num_mono = num_mono + 1 #suma en uno las aristas monocromáticas                          #adyacencia de ambas formas
                r = r+1 #Toma el nodo que sigue
    return num_mono #regresa el total de NAM



def calcula_NAM(Matriz,individuo,k):
    num_mono = 0
    for t in range(len(Matriz)-1):
        for j in range(t + 1, len(Matriz)-1):
            #if j== 90:
            #    print(":D!")
            #print("M[" + str(t) + "][" + str(j) + "]")
            if Matriz[t, j] == 1:
                if (str(t+1) in individuo and individuo[str(t+1)] > k - 1 or (str(j+1) in individuo and individuo[str(j+1)]> k-1)):
                    num_mono = num_mono + 1  # suma en uno las aristas monocromáticas
                    #print("M[" + str(t+1) + "][" + str(j+1) + "]="+str(t+1)+":" + str(individuo[str(t+1)]))
    return num_mono  # regresa el total de NAM


def main():
    for h in range(10):
        #if h == 2:
        #    print("ya!")
        archivosmall = open("k-small","a")
        carpeta = 'Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
        nombre = '/G' + str(numNodosAle) + '_' + str(probabilidad) + '_' + str(h + 1)
        #nombre = '/G' + str(probabilidad) + '_' + str(h + 1)
        Nombre_benchmark = carpeta + nombre
        #G = Lee_Grafo(Nombre_benchmark[h]) #Para benchmark
        G = Lee_Grafo(path + Nombre_benchmark) #Para Aleatorio
        d = nx.coloring.greedy_color(G, strategy=Estrategia)  # smallest_last independent_set
        col= max(d.values()) + 1
        #archivosmall.write(str(time.time() - start_time) + '\t' + str(col) + '\t')
        Matriz_Adyacencia = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()), dtype=int)
        numColores = max(d.values()) + 1
        for j in range(0, len(k)):
            start_time = time.time()
            d = nx.coloring.greedy_color(G, strategy=Estrategia)  # smallest_last independent_set
            archivosmall.write(str(time.time() - start_time) + '\t')
            Aristas_Mono = calcula_NAM(Matriz_Adyacencia, d, k[j])
            archivosmall.write(str(Aristas_Mono) + '\t')
            print(str(Aristas_Mono) + "\t" + str(col)+ "\t")
        archivosmall.write('\n')
        archivosmall.close()


if __name__ == '__main__':
    main()