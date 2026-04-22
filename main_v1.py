# Here we take care of the necessary imports, 
#   including the training, evaluation, and demo functions from the entrenamiento_v1 file
from entrenamiento_v1 import train, evaluate, demo

# The main function serves as the entry point of the program,
#   taking care of the training, evaluation, and demonstration of the agent.
def main():
    print("PRÁCTICA RL - VERSIÓN 1")
    print("Taxi-v3 básico con Q-learning\n")

    train()
    evaluate()
    demo()

if __name__ == "__main__":
    main()