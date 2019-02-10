# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:19:55 2018

@author: David
"""
import numpy as np 
import math
import itertools as itert
from scipy.spatial import distance
import matplotlib.pyplot as plt 
import random
import os





# FUNCTION TO DEFINE OMNIDIRECIONAL VARIOGRAM 
# ............................................................................................
def Variogram_experimental(x, y, cu, n_lags, lag_tol, lag):
    # ............ Calcular o variograma experimental ominidirecional ............

    # Calcular distancias 

    points = zip(x,y,cu)
    cordinates = zip(x,y)
    combinations_cordinates = list(itert.combinations(cordinates, 2))
    combinations_points = list(itert.combinations(points,2))
    distances = [distance.euclidean(i[0], i[1]) for i in combinations_cordinates]

    vectors = [lag*n for n in range(1, n_lags)]

    admissiveis_t = []
    for j in vectors:
        admissiveis = []
        for i, p in enumerate(distances):
            if p > (j - lag_tol) and p < (j + lag_tol):
                admissiveis.append((combinations_points[i][0][2],
                                    combinations_points[i][1][2]))
        admissiveis_t.append(admissiveis)

    experimental_variogram = []
    n_pares = []
    for i in admissiveis_t:
        soma = 0 
        par = 0 
        for p in range(len(i)):
            diferenca = (i[p][0] - i[p][1])**2
            soma += diferenca
            par += 1
        experimental_variogram.append(soma/(2*par))
        n_pares.append(par)

    plt.plot(vectors, experimental_variogram)
    return vectors, experimental_variogram, n_pares

# FUNCTION TO READ EXPERIMENTAL VARIOGRAMS 
# .................................................................................
    
def Read_experimental_variograms(filename):
    
    # Abrir o arquivo gslib e ler as linhas
    textfile = open(filename, "r")
    f = textfile.read().split("\n")
    
    # Determinar partição do arquivo
    g = []
    for i in f:
        g.append(i.split())
    
    # Teste se a última linha está vazia 
    g = [ x for x in g if x != []]
        
    
    variograms = [] 
    distances = []
    pares = []
    variogram = []
    distance = [] 
    par = []
    test = 0
    
    # Adicione os variogramas nas direções e as distancias 
    for i in g:
        if test == 0:
            if i[0] == "Semivariogram" or i[0] == "Variogram":
                variogram =[]
                distance = []
                par =[]
                test += 1
            else:
                if float(i[1]) != 0.000:
                    variogram.append(float(i[2]))
                    distance.append(float(i[1]))
                    par.append(int(i[3]))
        else: 
            if i[0] == "Semivariogram" or i[0] == "Variogram":
                variograms.append(variogram)
                distances.append(distance)
                pares.append(par)
                variogram = []
                distance = []
                par =[]
            else:
                if float(i[1]) != 0.000:
                    variogram.append(float(i[2]))
                    distance.append(float(i[1]))
                    par.append(int(i[3]))
    variograms.append(variogram)
    distances.append(distance)
    pares.append(par)
    return [j[1:] for j in variograms], [j[1:] for j in distances], [j[1:] for j in pares]
                
                
        
            

class Individuo:
    def __init__(self, vector, experimental_variogram, n_pares, modelo, regional, patamar = -999, fixed_nugget= -999, mutation_tax = 0.1):
        

        self.regional = regional
     
        
      
        self.fixed_nugget = fixed_nugget
        self.mutation_tax = mutation_tax
        self.patamar = patamar
        self.vector = vector
        self.experimental_variogram = experimental_variogram
        self.n_pares = n_pares
        self.nstruct = len(modelo)
        self.modelo = modelo
        self.range_var = np.array([max(self.vector)*(self.regional[i]-1 + np.random.random())  for i in range(self.nstruct)])
        self.geracao = 0
        self.nota = 0 
        vector2 =  np.array([np.random.random()*max(self.experimental_variogram) for i in range(self.nstruct)])
        self.contribution = vector2
        self.nugget = fixed_nugget
        
        if patamar != -999:
            soma_t = np.sum(self.contribution) 
            self.contribution = np.array(self.contribution)*float(self.patamar - self.nugget)/float(soma_t)
        
            

        if fixed_nugget == -999:
            vector3 = random.random()*max(self.contribution) 
            self.nugget = vector3



    def genes(self):
        Model = []
        for i in range(self.nstruct):
            model_struct = []
            for p, j in enumerate(self.vector):
                if self.modelo[i] == 0:
                    valor  =self.contribution[i]*(1.5*j/float(self.range_var[i]) -0.5*(j/float(self.range_var[i]))**3)
                    if j >= self.range_var[i]:
                        model_struct.append(self.contribution[i])
                    else:
                        model_struct.append(valor)
                if self.modelo[i] == 1:
                    model_struct.append(self.contribution[i]*(1-
                                        math.exp(-3.0*(j/float(self.range_var[i])**2))))
                if self.modelo[i] == 2:
                    model_struct.append(self.contribution[i]*(1-
                                        math.exp(-3.0*j/float(self.range_var[i]))))
            Model.append(model_struct)
        Model = np.array(Model)
        soma = np.array(Model.sum(axis=0))
        soma += self.nugget
        return soma
    
    def Pesos_distancia(self):
        pesos_dist =[]
        soma = 0
        for i in self.vector:
            soma += (1/float(i))
        for i in self.vector:
            pesos_dist.append((1/float(i))/soma)
        return pesos_dist
            
    def Fitness (self):
        modelo = self.genes()
        sqd = 0.000
        pesos_dist = self.Pesos_distancia()
        for p, i in enumerate(modelo):
            if self.patamar != -999:
                if i > self.patamar:
                    sqd += pesos_dist[p]*999999999
                else:
                    sqd += pesos_dist[p]*((i - self.experimental_variogram[p])**2/float(len(modelo)))
            else:
                sqd += pesos_dist[p]*((i - self.experimental_variogram[p])**2/float(len(modelo)))
        self.nota = sqd
        return sqd
    
    
    def Crossover(self, outro_individuo , vector, experimental_variogram, n_pares, nstruct, regional, fixed_nugget = -999, mutation_tax = 0.1):
        

        filho1 = Individuo(vector, experimental_variogram, n_pares, nstruct,self.regional, self.patamar, self.fixed_nugget, mutation_tax)
        filho2 = Individuo(vector, experimental_variogram, n_pares, nstruct, self.regional, self.patamar, self.fixed_nugget, mutation_tax)
        
        # receber as características do pai e da mae 


        filho1.contribution=self.contribution
        filho1.nugget=self.nugget 
        filho1.range_var=self.range_var
        filho1.geracao= self.geracao +1

        filho2.contribution=outro_individuo.contribution
        filho2.nugget=outro_individuo.nugget 
        filho2.range_var=outro_individuo.range_var
        filho2.geracao= outro_individuo.geracao +1


        
        random_struct = np.random.randint(0,(self.nstruct)-1)
        random_par = np.random.randint(0,2)
        armazenar_struc = 0 

        if self.patamar == -999:
            if random_par == 0: 
                armazenar_struc  = filho2.contribution[random_struct] 
                filho2.contribution[random_struct] = filho1.contribution[random_struct]
            if random_par == 1:
                armazenar_struc  = filho2.nugget
                filho2.nugget = filho1.nugget
            if random_par == 2: 
                armazenar_struc  =filho2.range_var[random_struct] 
                filho2.range_var[random_struct] = filho1.range_var[random_struct]
            
    
            if random_par == 0: 
                filho1.contribution[random_struct] = armazenar_struc
            if random_par == 1:
                filho1.nugget = armazenar_struc
            if random_par == 2: 
                filho1.range_var[random_struct] = armazenar_struc
        
            filho1.Fitness(), filho2.Fitness
        else: 
            
            if random_par == 0: 
                armazenar_struc  = filho2.contribution[random_struct] 
                filho2.contribution[random_struct] = filho1.contribution[random_struct]
                soma_t = np.sum(filho2.contribution) 
                filho2.contribution = filho2.contribution*float(self.patamar - filho2.nugget)/float(soma_t)
            if random_par == 1:
                if self.fixed_nugget == -999:
                    armazenar_struc  = filho2.nugget
                    filho2.nugget = filho1.nugget
                    soma_t = np.sum(filho2.contribution) 
                    filho2.contribution = filho2.contribution*float(self.patamar - self.nugget)/float(soma_t)
                else:
                    filho2.nugget = self.fixed_nugget
                    soma_t = np.sum(filho2.contribution) 
                    filho2.contribution = filho2.contribution*float(self.patamar - self.fixed_nugget)/float(soma_t)
            if random_par == 2: 
                armazenar_struc  =filho2.range_var[random_struct] 
                filho2.range_var[random_struct] = filho1.range_var[random_struct]
            
    
            if random_par == 0: 
                filho1.contribution[random_struct] = armazenar_struc
                soma_t = np.sum(filho1.contribution)
                filho1.contribution = filho1.contribution*float(self.patamar -self.nugget)/float(soma_t)
            if random_par == 1:
                if self.fixed_nugget == -999:
                    filho1.nugget = armazenar_struc
                    soma_t = np.sum(filho1.contribution)
                    filho1.contribution = filho1.contribution*float(self.patamar -self.fixed_nugget)/float(soma_t)
                else:
                    filho1.nugget = self.fixed_nugget
                    soma_t = np.sum(filho1.contribution)
                    filho1.contribution = filho1.contribution*float(self.patamar -self.nugget)/float(soma_t)
            if random_par == 2: 
                filho1.range_var[random_struct] = armazenar_struc
        
            filho1.Fitness(), filho2.Fitness
        return filho1, filho2
    
    def np_r(self):
        number = np.random.random()
        if number < 0.5:
            return 1
        else:
            return -1

    
    def Mutation(self):
        select = np.random.random()
        if select < self.mutation_tax: 
            random_struct1 = np.random.randint(0,(self.nstruct)-1)
            random_par1 = np.random.randint(0,2)
            if self.patamar != -999:
                if random_par1 == 0: 
                    
                    self.contribution[random_struct1] = self.contribution[random_struct1] + 0.05*self.np_r()*self.contribution[random_struct1]
                    soma_t = np.sum(self.contribution)
                    self.contribution= self.contribution*float(self.patamar - self.nugget)/float(soma_t)
                if random_par1 == 1:
                    if self.fixed_nugget == -999:
                        self.nugget = self.nugget + 0.05*self.np_r()*self.nugget
                        soma_t = np.sum(self.contribution) + self.nugget
                        self.nugget = self.nugget*float(self.patamar)/float(soma_t)
                        self.contribution = self.contribution*float(self.patamar)/float(soma_t)
                if random_par1 == 2: 
                    self.range_var[random_struct1] = self.range_var[random_struct1] + 0.05*self.np_r()*self.range_var[random_struct1]
            else:
                if random_par1 == 0: 
                    self.contribution[random_struct1] = self.contribution[random_struct1] + 0.05*self.np_r()*self.contribution[random_struct1]
                if random_par1 == 1:
                    if self.fixed_nugget == -999:
                        self.nugget = self.nugget + 0.05*self.np_r()*self.nugget
                if random_par1 == 2: 
                    self.range_var[random_struct1] = self.range_var[random_struct1] + 0.05*self.np_r()*self.range_var[random_struct1]
        self.Fitness()
    
        
class Genetic_Algorithm:
    
    def __init__(self, n_individuos, n_inter, seed):
        self.n_individuos = n_individuos 
        self.populacao = []
        self.notas = []
        self.geracao = 0 
        self.n_inter = n_inter
        if seed != -999:
            random.seed(10)
        

    def iniciar_populacao(self, vector, experimental_variogram, n_pares, nstruct,regional, patamar= -999, fixed_nugget= -999, mutation_tax = 0.1):
        
        

        for i in range(self.n_individuos):
            self.populacao.append(Individuo(vector, experimental_variogram, n_pares, nstruct,regional, patamar, fixed_nugget, mutation_tax))
        for i in self.populacao:
            i.Fitness()
            i.Mutation()
            self.notas.append(i.nota)
        self.melhor_solucao = [self.populacao[0].genes(), self.populacao[0].nota]
        self.melhor_individuo = self.populacao[0]
        self.ordena_populacao()
        
        
        
    def mutar_populacao(self):
        for i in self.populacao:
            i.Mutation()

    def ordena_populacao(self):
        lista1, lista2 = (list(t) for t in zip(*sorted(zip(self.notas, self.populacao))))
        self.notas = lista1
        self.populacao = lista2
        if self.melhor_solucao[1] > self.notas[0]:
            self.melhor_solucao = []
            self.melhor_solucao = [self.populacao[0].genes(), self.notas[0]]



            
    def seleciona_pai(self):
        comparison = [1/i for i in self.notas]
        comparison = np.cumsum(comparison)
        valor_sorteado = random.random()*max(comparison)
        soma = 0
        i = 0 
        for i, p in enumerate(comparison):
            soma += p
            if p >= valor_sorteado:
                return i 


    def nova_geracao(self,vector, experimental_variogram, n_pares, nstruct,regional, patamar = -999, fixed_nugget= -999, mutation_tax= 0.1):
        nova_geracao_v = []
        for i in range(int(self.n_individuos/2)):
            I_pai1, I_pai2 = self.seleciona_pai(), self.seleciona_pai()
            filho1, filho2 = self.populacao[I_pai1].Crossover(self.populacao[I_pai2],vector, experimental_variogram, n_pares, nstruct,fixed_nugget, mutation_tax )
            nova_geracao_v.append(filho1)
            nova_geracao_v.append(filho2)
        self.notas = []
        self.populacao = nova_geracao_v
        for i in nova_geracao_v:
            i.Fitness()
            self.notas.append(i.nota)      
        self.geracao += 1
        
        
    def otimizar(self, vector, experimental_variogram, n_pares, nstruct, regional, patamar = -999, fixed_nugget= -999, mutation_tax = 0.1, seed = -999):
        np.random.seed(seed)
        random.seed(seed)
        self.iniciar_populacao(vector, experimental_variogram, n_pares, nstruct,regional,patamar,fixed_nugget, mutation_tax)
        for i in range(self.n_inter):
            self.mutar_populacao()
            self.ordena_populacao()
            self.nova_geracao(vector, experimental_variogram, n_pares, nstruct,regional,patamar,fixed_nugget, mutation_tax)
            self.ordena_populacao()
        return self.melhor_solucao, self.populacao[0]

def Optimizing_variograms(textfile, list_of_structures,number_of_generations, number_of_individuals, regional, patamar=-999, fixed_nugget= -999,mutation_tax = 0.1, seed= -999):
    
    if regional == []:
        regional = [1.0 for i in range(0,len(list_of_structures))]
        
    
    variograms, distances, pares = Read_experimental_variograms(textfile)
    G= Genetic_Algorithm(number_of_individuals, number_of_generations, seed)
    
    results = []
    fixed_contribution =[]
    for i in range(len(variograms)):
        p, ind = G.otimizar(distances[i], variograms[i], pares[i], list_of_structures,regional, patamar, fixed_nugget, mutation_tax, seed)

        results =[]
        if i == 0:
            fixed_contribution = ind.contribution[:]
            results = ind.genes()
        else:
            ind.contribution = fixed_contribution
            results = ind.genes()
      
        
        
        print ("Variogram parameters")
        print ("............................................")
        print ("Contribution : " + str(ind.contribution))
        print ("Nugget : " +str(ind.nugget))
        print ("Range : "+str(ind.range_var))
        
    
        plt.title("Variogram")
        plt.plot(distances[i], results, label = "Model Dir = {}".format(str(i+1)))
        plt.plot(distances[i], variograms[i], marker= ".", label = "Experimental Dir ={}".format(str(i+1)))
        plt.xlabel("Distance (m)")
        plt.ylabel("Variogram")
        plt.legend()
    plt.show()
    


################### Input variables ############################
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# textfile = adress of gamv.out file 
# list of structures - structures in order - 0: spherical, 1: Exponential, 2: Gaussian
# regional - 

textfile = "gamv.out"
list_of_strucutures = [0,0]
regional = [1.0,1.5]
number_of_generations = 100
number_of_individuals = 400






Optimizing_variograms(textfile, list_of_strucutures,number_of_generations, number_of_individuals,regional,
                      patamar= 44.89,fixed_nugget=5.67, mutation_tax = 0.2, seed = 500)




    
