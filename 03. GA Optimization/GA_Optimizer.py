# GA_Optimizer - B115530
# The genetic algorithm object that deals with all the major optimisation operations.

import random
from keras.models import load_model
import hashlib
import numpy

class GeneticsAlgorithm:
    # Class Variables
    PopulationSize = None
    Predictor = None
    # Rates
    MutationRate =  None
    CrossOverRate = None
    # Data Holders
    Population = []
    FixedUser = []
    Date = []
    PossibleValues = []
    FitnessValues = []
    FitnessWheel = []

    def __init__(self, PopulationSize, PossibleValues, PathModel,User = [], Date = [], MutationRate = 0, CrossOverRate = 0):
        # Class Initization
        self.FixedUser = User
        self.Date = Date
        self.PossibleValues = PossibleValues
        self.PopulationSize = PopulationSize
        self.MutationRate = MutationRate
        self.CrossOverRate = CrossOverRate
        # self._GeneratePopulation()
        self.Predictor = load_model(PathModel)
	
# Function that takes care of creating the original population.
    def _GeneratePopulation(self):
        self.Population = []
        for i in range(self.PopulationSize):
            Solution = []
            for k in range(len(self.PossibleValues)):
                SelectedIndex = random.randint(0, len(self.PossibleValues[k]) - 1)
                Solution.append(self.PossibleValues[k][SelectedIndex])
            self.Population.append(Solution)

# Construct the ad and user characteristics in one array to be used for prediction.
    def _ConstructSolution(self, Chromosome):
        ArrayHash = Chromosome[:2] + self.FixedUser + Chromosome[2:]
        return ArrayHash

# Hashes the values to match with the same values the model was trained with.
    def _DataHasher(self, Value):
        return int(hashlib.sha1(str(Value).encode('utf-8')).hexdigest(), 16) % 10 ** 6

# Crossover operation.
    def CrossOver(self, Chromosome1, Chromosome2):
        if random.uniform(0,1)<=self.CrossOverRate:
            SplitIndex = random.randint(0, len(Chromosome1) - 1)
            Ind1 = Chromosome1[:SplitIndex] + Chromosome2[SplitIndex:]
            Ind2 = Chromosome2[:SplitIndex] + Chromosome1[SplitIndex:]
        else:
            Ind1 = Chromosome1
            Ind2 = Chromosome2
        return Ind1, Ind2

# Mutation Operation.
    def Mutate(self, Chromosome):
        for i, Cell in enumerate(Chromosome):
            if random.uniform(0,1) <= self.MutationRate:
                NewValue = self.PossibleValues[i][random.randint(0, len(self.PossibleValues[i]) - 1)]
                Chromosome[i] = NewValue
        return Chromosome

# Compute the fitness of the current solutions in the population.
    def ComputeFitnessList(self):
        FitnessList = []
        Predictors = []
        for i, Chromosome in enumerate(self.Population):
            Constructed = self._ConstructSolution(Chromosome)
            for i in range(len(Constructed)):
                Constructed[i] = self._DataHasher(Constructed[i])
            Predictors.append(Constructed + self.Date)
        Predictions = self.Predictor.predict_proba(numpy.array(Predictors))
        for k in Predictions:
            FitnessList.append(k[1])
        self.FitnessValues = FitnessList
        if len(self.FitnessValues) > self.PopulationSize:
            print("ERROR FITNESS LIST TOO LARGE.")

# Compute the roulette wheel values for selection.
    def ComputeRouletteWheel(self):
        FitnessSum = sum(self.FitnessValues)
        ProbabilitiesRes = []
        CumulativeProb = 0
        for Fitness in self.FitnessValues:
            CumulativeProb += Fitness/FitnessSum
            ProbabilitiesRes.append(CumulativeProb)
        self.FitnessWheel = ProbabilitiesRes
        if len(self.FitnessWheel) > self.PopulationSize:
            print("ERROR FITNESS WHEEL TOO LARGE.")

# Use the computed roulette wheel values to return two individuals for crossover and mutation.
    def SelectIndividuals(self):
        Value = random.uniform(0, 1)
        for i, Probability in enumerate(self.FitnessWheel):
            if Value < Probability:
                FirstIndividual = i
                break
        Value = random.uniform(0, 1)
        for i, Probability in enumerate(self.FitnessWheel):
            if Value < Probability:
                SecondIndividual = i
                break
        while FirstIndividual == SecondIndividual:
            Value = random.uniform(0, 1)
            for i, Probability in enumerate(self.FitnessWheel):
                if Value < Probability:
                    SecondIndividual = i
                    break
        return self.Population[FirstIndividual], self.Population[SecondIndividual]

# Update the population with a new generation;
    def UpdatePopulation(self, Population):
        self.Population = []
        self.Population = Population

# Update the parameters of the GA to make it ready for a new user. The function automatically creates
    # the new population;
    def UpdateParameters(self, MutationRate, CrossOverRate, User, Date, PopulationSize):
        # Class Initization
        self.FixedUser = User
        self.Date = Date
        self.PopulationSize = PopulationSize
        self.MutationRate = MutationRate
        self.CrossOverRate = CrossOverRate
        self.FitnessValues = []
        self.FitnessWheel = []
        self._GeneratePopulation()

# Print the list of parameters used.
    def GetParameters(self):
        print("------------------------------------")
        print("Mutation Rate: " + str(self.MutationRate))
        print("Crossover Rate: " + str(self.CrossOverRate))
        print("Population Size: " + str(self.PopulationSize))
        print("Fitness Values: " + str(self.FitnessValues))
        print("Fitness Wheel: " + str(self.FitnessWheel))
        print("------------------------------------")

# The optimization procedure. 
    def OptimizeSolution(self, Iterations):
        for i in range(Iterations):
            NextGeneration = []
            # Compute the fitness and the roulette wheel values for selection.
            self.ComputeFitnessList()
            self.ComputeRouletteWheel()
            # Keep selecting individuals until the size of the new generation is equal to the old one.
            for _ in range(int(self.PopulationSize / 2)):
                # Selection operation.
                Individual1, Individual2 = self.SelectIndividuals()
                # Crossover operation.
                Individual1, Individual2 = self.CrossOver(Individual1, Individual2)
                # Mutation operation.
                Individual1 = self.Mutate(Individual1)
                Individual2 = self.Mutate(Individual2)
                # Appending individuals to the new generation.
                NextGeneration.append(Individual1)
                NextGeneration.append(Individual2)
            # Update population.
            self.UpdatePopulation(NextGeneration)
        self.ComputeFitnessList()
        Average = numpy.mean(self.FitnessValues)
        Max = numpy.max(self.FitnessValues)
        BestSolution = self.Population[self.FitnessValues.index(Max)]
        return Average, Max, BestSolution
