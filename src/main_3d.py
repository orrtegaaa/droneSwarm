from entrenamiento_3d import train
from drone_env_3d import run_visual


def main():
    # entry point for the program
    # runs training and then the visual demonstration
    print("PRÁCTICA RL - VERSIÓN 3 (3D)")
    print("Entorno personalizado: DroneSwarm 3D  |  Servidor + 2 drones\n")

    # train the agent using Q-learning (independent Q-table per drone)
    server, hist = train()

    # shows visually what the model has learned
    # train is passed so the user can press R to retrain from the visual
    run_visual(server, hist, train_fn=train)


if __name__ == "__main__":
    # ensures that main() is executed when the file is run directly
    main()
