import environment
import agent
import numpy as np

class train_sys:
    def __init__(self,max_step = 200,
                 goal_step = 15,
                 episodes = 10000,
                 train_step=4,
                 save_step = 3,
                 ):
        self.episodes = episodes
        self.env = environment.Env(render_speed=0)
        self.env.reset(0)
        self.state = self.env.get_state()
        self.state_size = self.state.shape
        self.action_size = self.env.action_size
        self.action_space = self.env.action_space
        self.max_step = max_step
        self.goal_step = goal_step
        self.train_step = train_step
        self.save_step = save_step
        self.Agent = agent.PPO_Agent(state_shape=self.state_size,action_size=self.action_size)
        self.reward = 0
        self.end_count = 0
        self.target = 16


    def run(self):
        for epi in range(self.episodes):
            done = False
            self.reward = 0
            self.Agent.count = 0
            dir = None
            train_count = 0
            while not done:

                action, action_one_hot,action_prob = self.Agent.get_act(self.state[np.newaxis])
                if(action == 0):
                    dir = '←'

                elif(action == 1):
                    dir = '→'

                elif(action == 2):
                    dir = '↓'

                elif(action == 3):
                    dir = '↑'

                next_state, self.reward, done = self.env.step(action)
                self.Agent.add_buffer(self.state, action_one_hot, self.reward,next_state,action_prob,done,self.env.tot_reward,epi)
                print("log>> agent " + dir + " " + str(self.env.tot_reward) + " " + str(self.reward))


                self.state = next_state
                self.Agent.count = self.env.steps


                if(self.Agent.count > self.max_step):
                    print("log>> over 200 step, break now!")
                    break

                if(len(self.Agent.s) >= 64):
                    self.Agent.train()
                    self.Agent.save_model()
                    train_count += 1
                    if (epi >= 10 and train_count >= 3):
                        train_count = 0
                        if self.evaluate():
                            self.Agent.writer_save()
                            return 0


            print("Episode >> " +str(epi)+" reward("+str(self.env.tot_reward)+")")

            self.env.reset(epi + 1)

    def evaluate(self):
        print("log // evaluate model start")
        sum_steps = 0
        next_state = None
        for i in range(5):
            done = False
            self.env.reset(epi=i)
            state = self.env.get_state()
            next_state = None
            step = 0
            dir = None
            while not done:
                step += 1
                action = self.Agent.evaluate_get_act_(state[np.newaxis])
                if (action == 0):
                    dir = '←'

                elif (action == 1):
                    dir = '→'

                elif (action == 2):
                    dir = '↓'

                elif (action == 3):
                    dir = '↑'
                next_state, _, done = self.env.step(action)
                print("log>> agent " + str(dir) + " " + str(self.env.tot_reward))
                state = next_state
                if (step > 100):
                    print("log // test fail")
                    return False
            score = self.env.tot_reward
            print("log // epi " + str(i) + " >> steps : " + str(step) + " / score : " + str(score))

            sum_steps += step

            if (score < 0):
                print("log // test fail")
                return False

        avg_steps = sum_steps / 5

        if (avg_steps > 8):
            print("log // test pass!")

            return True

        else:
            return False
#
# class train_multi_sys:
#     def __init__(self,max_step = 300,
#                  goal_step = 15,
#                  episodes = 10000,
#                  train_step=4,
#                  save_step = 5,
#                  ):
#         self.episodes = episodes
#         self.env = environment.Env(render_speed=0)
#         self.env.reset(0)
#         self.state = self.env.get_state()
#         self.state_size = self.state.shape
#         self.action_size = self.env.action_size
#         self.action_space = self.env.action_space
#         self.max_step = max_step
#         self.goal_step = goal_step
#         self.train_step = train_step
#         self.save_step = save_step
#         self.Agent = agent.PPO_Agent(state_shape=self.state_size,action_size=self.action_size)
#         self.reward = 0
#         self.end_count = 0
#
#         self.num_agents = 8
#
#     def run(self):
#
#         agents,parent,child = [],[],[]
#
#         for i in range(self.num_agents):
#             parent,child = Pipe()
#             agent =


class test_sys:
    def __init__(self):
        self.env = environment.Env(render_speed=0.5)
        self.env.reset(0)
        self.state = self.env.get_state()
        self.state_size = self.state.shape
        self.action_size = self.env.action_size
        self.action_space = self.env.action_space
        self.Agent = agent.PPO_Agent(state_size=self.state_size,action_size=self.action_size,action_space=self.action_space)
        self.reward = 0

    def run(self):
        done = False
        self.reward = 0
        self.Agent.count = 0
        while not done:

            action,action_one_hot = self.Agent.get_act(self.state)

            print("log>> agent "+str(action)+" "+str(self.env.reward)+" "+str(self.Agent.steps))
            next_state,self.reward,done= self.env.step(action)
            self.Box_Agent.add_memory(self.state, action_one_hot, self.reward, done)
            self.state = next_state
            self.Agent.count += 0


            print("Episode >> reward("+str(self.env.reward)+")")



if(__name__ == "__main__"):
    train = train_sys()
    _ = input("press any key to start")
    _ = train.run()




