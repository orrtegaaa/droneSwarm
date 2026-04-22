from entrenamiento_v3 import train, evaluate, demo

def main():
    # entry point for the program
    # runs training, evaluation and finally the visual demonstration
    print("PRÁCTICA RL - VERSIÓN 3")
    print("Entorno personalizado: DroneSwarm\n")

    #train the agent using Q-learning
    train()

    #evaluation of the policy earned by the episodes
    evaluate()

    #shows visually what the model has learned
    demo()

if __name__ == "__main__":
    #ensures that main() is executed when the file is run directly
    main()