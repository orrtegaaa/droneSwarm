from wrappers import make_env_v2


def main():
    env = make_env_v2()
    obs, info = env.reset()

    print("Versión 2 - Taxi-v3 con wrappers")
    print("Observación inicial:", obs)

    env.close()


if __name__ == "__main__":
    main()
