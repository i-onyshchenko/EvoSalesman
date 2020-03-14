#  by Ihor Onyshchenko

from Pyro4 import expose
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

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
                pathDistance += fromCity.distance(toCity)
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
            res += x

        return res

    @staticmethod
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
    def mutate(individual, mutationRate):
        for swapped in range(len(individual)):
            if (random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    @staticmethod
    @expose
    def mutateChunk(self, individuals, mutationRate):
        return [Solver.mutate(ind, mutationRate) for ind in individuals]

    def breedPopulation(self, matingpool, eliteSize):
        children = [matingpool[i] for i in range(eliteSize)]
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        parents = [[pool[i], pool[len(matingpool) - i - 1]] for i in range(length)]
        chunks = np.split(parents, len(self.workers))

        mapped = []
        for i in range(len(self.workers)):
            mapped.append(self.workers[i].breedChunk(chunks[i]))

        return children + self.myreduce(mapped)

    def mutatePopulation(self, population, mutationRate):
        chunks = np.split(population, len(self.workers))

        mapped = []
        for i in range(len(self.workers)):
            mapped.append(self.workers[i].mutateChunk(chunks[i], mutationRate))

        mutatedPop = self.myreduce(mapped)
        return mutatedPop

    def nextGeneration(self, currentGen, eliteSize, mutationRate):
        popRanked = Solver.rankRoutes(currentGen)
        selectionResults = Solver.selection(popRanked, eliteSize)
        matingpool = Solver.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool, eliteSize)
        nextGeneration = self.mutatePopulation(children, mutationRate)
        return nextGeneration

    def geneticAlgorithm(self, population, popSize, eliteSize, mutationRate, generations):
        pop = Solver.initialPopulation(popSize, population)
        print("Initial distance: " + str(1 / Solver.rankRoutes(pop)[0][1]))

        for i in range(generations):
            pop = self.nextGeneration(pop, eliteSize, mutationRate)

        print("Final distance: " + str(1 / Solver.rankRoutes(pop)[0][1]))
        bestRouteIndex = Solver.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        return bestRoute

    def solve(self):
        self.read_input()
        bestRoute = self.geneticAlgorithm(self.cities, self.popSize, self.eliteSize, self.mutationRate, self.generations)
        self.write_output(bestRoute)

    def read_input(self):
        pass

    def write_output(self, route):
        pass

    def geneticAlgorithmPlot(self, population, popSize, eliteSize, mutationRate, generations):
        pop = Solver.initialPopulation(popSize, population)
        progress = [1 / Solver.rankRoutes(pop)[0][1]]

        for i in range(generations):
            pop = self.nextGeneration(pop, eliteSize, mutationRate)
            progress.append(1 / Solver.rankRoutes(pop)[0][1])

        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()


if __name__ == "__main__":
    cityList = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for i in range(25)]

    # geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
    Solver().geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
