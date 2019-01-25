import random

'''Encuentra la bolsa con la máxima cardinalidad y regresa el índice donde la bolsa'''
def Max_cadinalidad(individuo):
    ind=max(individuo, key=lambda k: len(individuo[k])) #encuentra el índice donde encuentra la cardinalidad máxima
    return ind


'''Elimina los nodos del nuevo individuo del individuo del cual NO obtuvo los nodos'''
def quita_bolsa(Bolsa, individuo, numColores):
    for valor in list(Bolsa):   #recorre la lista de nodos de la bolsa que se ingresaron en el nuevo individuo
        for i in range(numColores): #recorre todas las bolsas del individuo
             if valor in individuo[i]:  #pregunta si el nodo del nuevo individuo se encuentra en la bolsa i
                del individuo[i][valor] #Si el valor se encuentra en la bolsa, lo elimina
                break                   #y termina el ciclo


'''Recibe dos individuos, genera la cruza mediante una estrategia greedy de cardinalidad máxima
y forma un nuevo individuo'''
def GPX(individuos, numColores):
    nuevo_individuo = {}
    for l in range(numColores): #recorre las bolsas de colores
        if l%2 == 0:  #toma a los padres por turnos, para que sea balanceado
            ind = Max_cadinalidad(individuos[0])    #obtiene el índice de la bolsa con cardinalidad máxima del padre 1
            nuevo_individuo[l] = dict(individuos[0][ind]) #ingresa la bolsa al nuevo individuo
            quita_bolsa(individuos[0][ind], individuos[1], numColores)    #elimina los nodos en el padre 2
            individuos[0][ind].clear()  #elimina la bolsa del padre 1
        else:
            ind = Max_cadinalidad(individuos[1]) #obtiene el índice de la bolsa con cardinalidad máxima del padre 2
            nuevo_individuo[l] = dict(individuos[1][ind]) #ingresa la bolsa al nuevo individuo
            quita_bolsa(individuos[1][ind], individuos[0], numColores)  #elimina los nodos en el padre 1
            individuos[1][ind].clear()  #elimina la bolsa del padre 2
    for l in range(numColores): #Recorre nuevamente las bolsas de colores
        if individuos[0][l]:    #checa si la bolsa está vacía
            for nodo in individuos[0][l].keys():    #toma cada valor en la bolsa
                nuevo_individuo[random.randint(0, numColores-1)][nodo]=nodo #Lo inserta en una bolsa aleatoria
    return nuevo_individuo  #Retorna el nuevo individuo
