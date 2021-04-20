import gym
#from gym.envs.registration import register
#import readchar

env = gym.make("CartPole-v0")
print(env)
obs = env.reset()
print(obs)  # 수평위치(0.0=중앙), 속도, 막대의각도(0.0=수직), 각속도

img = env.render(mode='rgb_array')  # 환경출력
print(img.shape)
