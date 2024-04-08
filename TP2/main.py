from engine import GeneticAlgorithmEngine
from arguments import ProgramArguments

if __name__ == '__main__':
    pArguments = ProgramArguments()
    
    gaEngine = GeneticAlgorithmEngine(arguments=pArguments.arguments)

    gaEngine.start()