#!/usr/bin/env python3.6
#  by Ihor Onyshchenko


from Pyro4 import expose
import numpy as np
import pandas as pd
import random
import operator


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def cityDistance(self, city1, city2):
        xDis = abs(city1[0] - city2[0])
        yDis = abs(city1[1] - city2[1])
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += self.cityDistance(fromCity, toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.cities = None
        self.popSize = None
        self.eliteSize = None
        self.mutationRate = None
        self.generations = None
        print("Inited")

    @staticmethod
    def createRoute(cityList):
        route = random.sample(cityList, len(cityList))
        return route

    @staticmethod
    def initialPopulation(popSize, cityList):
        population = [Solver.createRoute(cityList) for _ in range(popSize)]

        return population

    @staticmethod
    def rankRoutes(population):
        fitnessResults = {}
        for i in range(len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitness()
        return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

    @staticmethod
    def selection(popRanked, eliteSize):
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        selectionResults = [popRanked[i][0] for i in range(eliteSize)]

        for _ in range(len(popRanked) - eliteSize):
            pick = 100 * random.random()
            for i in range(len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break

        return selectionResults

    @staticmethod
    def matingPool(population, selectionResults):
        matingpool = [population[selectionResults[i]] for i in range(len(selectionResults))]

        return matingpool

    @staticmethod
    def myreduce(mapped):
        res = []
        for x in mapped:
            res = res + x.value

        return res

    @staticmethod
    @expose
    def breed(parent1, parent2):
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        childP1 = [parent1[i] for i in range(startGene, endGene)]
        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child

    @staticmethod
    @expose
    def breedChunk(parents):
        return [Solver.breed(*par) for par in parents]

    @staticmethod
    @expose
    def mutate(individual, mutationRate):
        for swapped in range(len(individual)):
            if random.random() < mutationRate:
                swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    @staticmethod
    @expose
    def mutateChunk(individuals, mutationRate):
        return [Solver.mutate(ind, mutationRate) for ind in individuals]

    def breedPopulation(self, matingpool, eliteSize):
        children = [matingpool[i] for i in range(eliteSize)]
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        parents = [[pool[i], pool[len(matingpool) - i - 1]] for i in range(length)]
        chunks = np.split(np.array(parents), len(self.workers))
        chunks = [chunk.tolist() for chunk in chunks]

        mapped = [self.workers[i].breedChunk(chunks[i]) for i in range(len(self.workers))]

        return children + self.myreduce(mapped)

    def mutatePopulation(self, population, mutationRate):
        chunks = np.split(np.array(population), len(self.workers))
        chunks = [chunk.tolist() for chunk in chunks]

        mapped = [self.workers[i].mutateChunk(chunks[i], mutationRate) for i in range(len(self.workers))]

        return self.myreduce(mapped)

    def nextGeneration(self, currentGen, eliteSize, mutationRate):
        popRanked = self.rankRoutes(currentGen)
        selectionResults = self.selection(popRanked, eliteSize)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool, eliteSize)
        nextGeneration = self.mutatePopulation(children, mutationRate)
        return nextGeneration

    def geneticAlgorithm(self, population, popSize, eliteSize, mutationRate, generations):
        pop = self.initialPopulation(popSize, population)
        print("Initial distance: " + str(1 / self.rankRoutes(pop)[0][1]))

        for i in range(generations):
            pop = self.nextGeneration(pop, eliteSize, mutationRate)

        print("Final distance: " + str(1 / self.rankRoutes(pop)[0][1]))
        bestRouteIndex = self.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        return bestRoute

    def solve(self):
        self.read_input()
        bestRoute = self.geneticAlgorithm(self.cities, self.popSize, self.eliteSize, self.mutationRate, self.generations)
        self.write_output(bestRoute)

    def read_input(self):
        with open(self.input_file_name, 'r+') as f:
            lines = f.readlines()
            num_cities = int(lines[0])
            self.cities = [list(map(lambda x: int(x), lines[i].split(','))) for i in range(1, num_cities+1)]
            self.popSize = int(lines[num_cities+1])
            self.eliteSize = int(lines[num_cities+2])
            self.mutationRate = float(lines[num_cities+3])
            self.generations = int(lines[num_cities+4])

    def write_output(self, route):
        with open(self.output_file_name, 'w+') as f:
            route = "->".join(list(map(str, route)))
            f.write(route)


if __name__ == "__main__":
    cityList = [[int(random.random() * 200), int(random.random() * 200)] for i in range(25)]

    Solver([Solver(), Solver()], 'input.txt', 'output.txt').solve()
