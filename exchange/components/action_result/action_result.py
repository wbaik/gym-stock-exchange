from dataclasses import dataclass


@dataclass
class ActionResult:
    '''
    From the doc, https://gym.openai.com/docs/, `env.step(action)` is expected to return the followings:

        observation (object): an environment-specific object representing your observation of the environment.
                              For example, pixel data from a camera, joint angles and joint velocities of a robot,
                              or the board state in a board game.
        reward (float): amount of reward achieved by the previous action. The scale varies between environments, but
                        the goal is always to increase your total reward.
        done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up
                        into well-defined episodes, and done being True indicates the episode has terminated.
                        (For example, perhaps the pole tipped too far, or you lost your last life.)
        info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning
                     (for example, it might contain the raw probabilities behind the environment’s last state change).
                     However, official evaluations of your agent are not allowed to use this for learning.
    '''
    subsequent_price: float
    reward: float
    done: bool
    info: dict
