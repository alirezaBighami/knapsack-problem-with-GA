import random
import sys
import operator
from matplotlib import pyplot as plt


class Knapsack(object):

    # run the Genetic Algorithm
    def geneticAlgorithm(self):
        self.selection()
        newGenes = []
        pop = len(self.bestPops) - 1
        for i in range(0, pop):
            if i < pop - 1:
                temp1 = self.bestPops[0]
                temp2 = self.bestPops[1]
                nchild1, nchild2 = self.funcCrossover(temp1, temp2, self.coRate)
                newGenes.append(nchild1)
                newGenes.append(nchild2)
            else:
                temp1 = self.bestPops[1]
                temp2 = self.bestPops[0]
                nchild1, nchild2 = self.funcCrossover(temp1, temp2, self.coRate)
                newGenes.append(nchild1)
                newGenes.append(nchild2)

        # mutate the new children and potential parents to ensure global optima found
        for i in range(len(newGenes)):
            newGenes[i] = self.funcMutation(newGenes[i], self.muRate)
            self.fitnesses.append(str(self.fitness(newGenes[i])))

        self.iterated += 1
        self.genes = newGenes
        self.bests = []
        self.bestPops = []

        newGenesNumber = self.initiPopulation // 2
        index = 0
        while(index < newGenesNumber):
            geneFitness = self.fitness(newGenes[index])
            if self.maxFitness < geneFitness:
                self.maxFitness = geneFitness
                self.maxBag = "".join(str(newGenes[index]))
                self.foundIterated = self.iterated
            index += 1


        # if stop method return True means terminate Condition is met
        if Knapsack.stop(self):
            print("optimal ", self.maxBag, " with fitness: ", self.maxFitness, " found in ",
                  self.foundIterated, "generations")

            intFitnesses = list(map(int, self.fitnesses))
            index = 0
            range1 = (self.foundIterated - 1) * self.initiPopulation // 2 -1
            while (index < range1):
                if intFitnesses[index] == -1 and index != 0:
                    intFitnesses[index] = intFitnesses[index - 1]
                index += 1

            index = 0
            bestChileds = []
            while (index < range1):
                index1 = index + 1
                best = intFitnesses[index]
                bestChileds.append(intFitnesses[index])
                while (index1 < index + self.initiPopulation // 2):
                    if intFitnesses[index1] > best:
                        best = intFitnesses[index1]
                        bestChileds[-1] = best
                    index1 += 1
                index += self.initiPopulation // 2
            # self.fitnesses.append(self.maxFitness)
            if (bestChileds[-1] != self.maxFitness):
                bestChileds.append(self.maxFitness)
            sizeOfNemodar = len(bestChileds)
            # Create the chart of fitnesses
            plt.scatter(range(2, sizeOfNemodar + 2), bestChileds[0:sizeOfNemodar], c="red")
            plt.plot(range(2, sizeOfNemodar + 2), bestChileds[0:sizeOfNemodar], "red")
            plt.legend(["nemodar"])
            plt.xlabel("iteration")
            plt.ylabel("fitness")
            plt.show()
        # run the algorithm again
        else:
            self.geneticAlgorithm()

    # create the initial population
    def createGenes(self):
        for i in range(self.initiPopulation):
            gene = []
            for k in range(0, self.initiPopulation):
                k = random.randint(0, 1)
                gene.append(k)
            self.genes.append(gene)
        print("Initial genes : \n", self.genes)

    # initialize variables  and set the properties
    def properties(self, weights, profits, capacity, initiPopulation, muRate, coRate):
        self.weights = weights
        self.profits = profits
        self.capacity = capacity
        self.initiPopulation = initiPopulation
        self.muRate = muRate
        self.coRate = coRate
        self.newGenes = []
        self.genes = []
        self.bests = []
        self.bestPops = []
        self.iterated = 1
        self.rwArr = []
        self.fitnesses = []
        self.maxFitness = 0
        self.foundIterated = 0
        self.maxBag = ""
        self.createGenes()

        # increase max recursion for long stack
        iMaxStackSize = 15000
        sys.setrecursionlimit(iMaxStackSize)

    # calculate the fitness and if it's capacity< , return -1
    def fitness(self, item):
        sumOfWeights = 0
        sumOfProfits = 0
        for index, i in enumerate(item):
            if i == 0:
                continue
            else:
                sumOfWeights += self.weights[index]
                sumOfProfits += self.profits[index]

        # if greater than the optimal return -1 or the number otherwise
        if sumOfWeights > self.capacity:
            return -1
        else:
            return sumOfProfits

    # run generations of Genetic Algorithm
    def selection(self):
        # loop through parents and calculate fitness
        bestPop = self.initiPopulation // 2
        for i in range(len(self.genes)):
            gene = self.genes[i]
            fitness = self.fitness(gene)
            self.bests.append((fitness, gene))
        # sort the fitness list by fitness
        self.bests.sort(key=operator.itemgetter(0), reverse=True)

        # roulette-wheel
        indexes = 0
        while (indexes < bestPop):
            sumOfAllFitesses = 0
            indexes += 1
            atPoint = 1

            # for calculate sumOfAllFitnesses
            for item in self.bests:
                if item[0] != -1:
                    sumOfAllFitesses += item[0]

            # for calculate rwArr for the range of each parent
            self.rwArr.clear()
            for item in self.bests:
                if item[0] != -1:
                    if sumOfAllFitesses > 0:
                        self.rwArr.append(item[0] / sumOfAllFitesses)
                    else:
                        self.rwArr.append(0)

            randomNumber = random.uniform(0, 1)
            sum = 0
            index = -1
            for item in self.rwArr:
                index += 1
                sum += item
                if randomNumber < sum and atPoint == 1:
                    self.bestPops.append(self.bests[index][1])
                    self.bests.pop(index)
                    atPoint = 0
        while (len(self.bestPops) < bestPop):
            self.bestPops.append(self.bests[0][1])
            self.bests.pop(0)

    # crossover two parents to produce two children by miixing them under random ration each time
    def funcCrossover(self, child1, child2, coRate):
        randomFloat = random.uniform(0, 1)
        if randomFloat > coRate:
            threshold = random.randint(1, len(child1) - 1)
            tmp1 = child1[threshold:]
            tmp2 = child2[threshold:]
            child1 = child1[:threshold]
            child2 = child2[:threshold]
            child1.extend(tmp2)
            child2.extend(tmp1)
        return child1, child2

    # mutate children after certain condition
    def funcMutation(self, child, muRate):
        for i in range(len(child)):
            random_muRate = random.uniform(0, 1)
            if random_muRate < muRate:
                if child[i] == 1:
                    child[i] = 0
                else:
                    child[i] = 1
        return child

    # stop if more than 100 * initiPopulation generated
    # terminate Condition
    def stop(self):
        if self.iterated == 100 * self.initiPopulation:
            return True
        else:
            return False

# give inputs from user and run the code
def giveInputs():
    weights       = []
    profits       = []
    inputFilePath = input("Enter input name: ")
    # inputFilePath = "./input.txt"
    file          = open(inputFilePath, "r")
    inputString   = file.read()
    inputArray    = inputString.split()
    initiPopulation    = int(inputArray[0])
    muRate = float(inputArray[1])
    coRate        = float(inputArray[2])
    coRate        = coRate / 100
    capacity      = int(inputArray[3])
    index         = 0
    while (index < initiPopulation):
        weights.append(int(inputArray[index + 4]))
        profits.append(int(inputArray[index + 4 + initiPopulation]))
        index += 1
    k = Knapsack()
    k.properties(weights, profits, capacity, initiPopulation, muRate, coRate)
    k.geneticAlgorithm()

giveInputs()
