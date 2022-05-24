import time
import yaml
import numpy as np
from trainer import Trainer
from evaluator import Evaluator


def main():
    start = time.time()  # time start
    # load config file
    config_file_name = 'config.yaml'
    with open(config_file_name) as file:
        config = yaml.safe_load(file)
    trainer = Trainer(config)
    trainer.reset()
    evaluator = Evaluator(config)
    evaluator.reset()

    config = config['main']
    # collect episode
    obs, rew, act = trainer.step(config['collect_episode_size'])
    trainer.add_to_buffer(obs, act, rew)

    train_steps = config['train_steps']
    for iter in range(train_steps):
        print(f"-------------------train_step: {iter}-------------------")
        # train Dynamics and Behavior
        observ, action, reward = trainer.sample_buffer()
        trainer.train(observ, action, reward, config['collect_interval'])

        # environment interaction
        obs, rew, act = trainer.step(config['interaction_steps'])
        print("environment interaction")
        print(f"observ : {np.argmax(obs, axis=-1)}")
        print(f"action : {np.argmax(act, axis=-1)}")
        trainer.add_to_buffer(obs, act, rew)

        # # test step
        # if (iter + 1) % 10 == 0:
        #     obs, rew, act = evaluator.step(100)
        #     print("test step")
        #     print(f"observ : {np.argmax(obs, axis=-1)}")
        #     print(f"action : {np.argmax(act, axis=-1)}")
        #     print(f"reward : {np.round(rew)}")

        trainer.save_train_metrics()
        evaluator.save_result()

    print(f"time :{time.time() - start:.1f}")  # print runtime
    print("fin")


if __name__ == '__main__':
    np.set_printoptions(threshold=100000, linewidth=100000)
    main()
