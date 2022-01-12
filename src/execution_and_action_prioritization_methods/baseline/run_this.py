from env import Env
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

#####################  hyper parameters  ####################
CHECK_EPISODE = 4
LEARNING_MAX_EPISODE = 25 #10
MAX_EP_STEPS = 1500 #600 #3000
TEXT_RENDER = False
SCREEN_RENDER = False
CHANGE = False
SLEEP_TIME = 0.1

#####################  function  ####################
def exploration (a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ####################################

if __name__ == "__main__":
    env = Env()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location, MAX_REQ_TIMER, ALGORITHM, METHOD, CONCEPT, SERVER_LIMIT_RANGE, LATENCY_REQUIREMENTS = env.get_inf()
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    b_var = 1
    ep_reward = []
    ep_reward_task_prio_1 = []
    ep_reward_task_prio_2= []
    ep_reward_task_prio_3 = []
    ep_reward_task_lat_1 = []
    ep_reward_task_lat_2= []
    ep_reward_task_lat_3 = []
    ep_reward_app_typ_1 = []
    ep_reward_app_typ_2 = []
    ep_reward_app_typ_3 = []
    ep_reward_app_typ_4 = []
    ep_penalization = []
    ep_penalization_task_prio_1 = []
    ep_penalization_task_prio_2 = []
    ep_penalization_task_prio_3 = []
    ep_penalization_task_lat_1 = []
    ep_penalization_task_lat_2 = []
    ep_penalization_task_lat_3 = []
    ep_penalization_app_typ_1 = []
    ep_penalization_app_typ_2 = []
    ep_penalization_app_typ_3 = []
    ep_penalization_app_typ_4 = []
    ep_usage_history = []
    ep_security_requirement = []
    r_v, b_v = [], []
    var_reward = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []
    while var_counter < LEARNING_MAX_EPISODE:
        print("Learning step is: ", var_counter, "/", LEARNING_MAX_EPISODE)
        #print("episode: ", episode)
        # initialize
        s = env.reset()
        ep_reward.append(0)
        ep_reward_task_prio_1.append(0)
        ep_reward_task_prio_2.append(0)
        ep_reward_task_prio_3.append(0)
        ep_reward_task_lat_1.append(0)
        ep_reward_task_lat_2.append(0)
        ep_reward_task_lat_3.append(0)
        ep_reward_app_typ_1.append(0)
        ep_reward_app_typ_2.append(0)
        ep_reward_app_typ_3.append(0)
        ep_reward_app_typ_4.append(0)
        ep_penalization.append(0)
        ep_penalization_task_prio_1.append(0)
        ep_penalization_task_prio_2.append(0)
        ep_penalization_task_prio_3.append(0)
        ep_penalization_task_lat_1.append(0)
        ep_penalization_task_lat_2.append(0)
        ep_penalization_task_lat_3.append(0)
        ep_penalization_app_typ_1.append(0)
        ep_penalization_app_typ_2.append(0)
        ep_penalization_app_typ_3.append(0)
        ep_penalization_app_typ_4.append(0)     
        ep_usage_history.append(0)   
        ep_security_requirement.append(0) 
        if SCREEN_RENDER:
            env.initial_screen_demo()

        for j in range(MAX_EP_STEPS):
            time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # DDPG
            # choose action according to state
            a = ddpg.choose_action(s)  # a = [R B O]
            # add randomness to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # store the transition parameter
            s_, r,  p, r_task_prio_1, r_task_prio_2, r_task_prio_3, p_task_prio_1, p_task_prio_2, p_task_prio_3, r_task_lat_1, r_task_lat_2, r_task_lat_3, p_task_lat_1, p_task_lat_2, p_task_lat_3, r_app_typ_1, r_app_typ_2, r_app_typ_3, r_app_typ_4, p_app_typ_1, p_app_typ_2, p_app_typ_3, p_app_typ_4, uh, sr = env.ddpg_step_forward(a, r_dim, b_dim)
            ddpg.store_transition(s, a, r / 10, s_)
            # learn
            if ddpg.pointer == ddpg.memory_capacity:
                print("start learning")
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            s = s_
            # sum up the reward
            ep_reward[episode] += r
            ep_reward_task_prio_1[episode] += r_task_prio_1
            ep_reward_task_prio_2[episode] += r_task_prio_2
            ep_reward_task_prio_3[episode] += r_task_prio_3
            ep_reward_task_lat_1[episode] += r_task_lat_1
            ep_reward_task_lat_2[episode] += r_task_lat_2
            ep_reward_task_lat_3[episode] += r_task_lat_3
            ep_reward_app_typ_1[episode] += r_app_typ_1
            ep_reward_app_typ_2[episode] += r_app_typ_2
            ep_reward_app_typ_3[episode] += r_app_typ_3
            ep_reward_app_typ_4[episode] += r_app_typ_4
            ep_usage_history[episode] = uh
            ep_security_requirement[episode] = sr
            # sum up the penalization
            ep_penalization[episode] += p
            ep_penalization_task_prio_1[episode] += p_task_prio_1
            ep_penalization_task_prio_2[episode] += p_task_prio_2
            ep_penalization_task_prio_3[episode] += p_task_prio_3
            ep_penalization_task_lat_1[episode] += p_task_lat_1
            ep_penalization_task_lat_2[episode] += p_task_lat_2
            ep_penalization_task_lat_3[episode] += p_task_lat_3
            ep_penalization_app_typ_1[episode] += p_app_typ_1
            ep_penalization_app_typ_2[episode] += p_app_typ_2
            ep_penalization_app_typ_3[episode] += p_app_typ_3
            ep_penalization_app_typ_4[episode] += p_app_typ_4
            # in the end of the episode
            if j == MAX_EP_STEPS - 1:
                var_reward.append(ep_reward[episode])
                r_v.append(r_var)
                b_v.append(b_var)
                print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], ' Reward for tasks with priority 1: %5d' % ep_reward_task_prio_1[episode], ' Reward for tasks with priority 2: %5d' % ep_reward_task_prio_2[episode], ' Reward for tasks with priority 3: %5d' % ep_reward_task_prio_3[episode], ' Reward for tasks with latency 1: %5d' % ep_reward_task_lat_1[episode], ' Reward for tasks with latency 2: %5d' % ep_reward_task_lat_2[episode], ' Reward for tasks with latency 3: %5d' % ep_reward_task_lat_3[episode], ' Reward for tasks with application type 1: %5d' % ep_reward_app_typ_1[episode], ' Reward for tasks with application type 2: %5d' % ep_reward_app_typ_2[episode], ' Reward for tasks with application type 3: %5d' % ep_reward_app_typ_3[episode], ' Reward for tasks with application type 4: %5d' % ep_reward_app_typ_4[episode], ' Penalization: %5d' % ep_penalization[episode], ' Penalization for tasks with priority 1: %5d' % ep_penalization_task_prio_1[episode], ' Penalization for tasks with priority 2: %5d' % ep_penalization_task_prio_2[episode], ' Penalization for tasks with priority 3: %5d' % ep_penalization_task_prio_3[episode], ' Penalization for tasks with latency 1: %5d' % ep_penalization_task_lat_1[episode], ' Penalization for tasks with latency 2: %5d' % ep_penalization_task_lat_2[episode], ' Penalization for tasks with latency 3: %5d' % ep_penalization_task_lat_3[episode], ' Penalization for tasks with application type 1: %5d' % ep_penalization_app_typ_1[episode], ' Penalization for tasks with application type 2: %5d' % ep_penalization_app_typ_2[episode], ' Penalization for tasks with application type 3: %5d' % ep_penalization_app_typ_3[episode], ' Penalization for tasks with application type 4: %5d' % ep_penalization_app_typ_4[episode], ' Usage history: ', ep_usage_history[episode], ' Security requirement: %5d' % ep_security_requirement[episode], '###  r_var: %.2f ' % r_var,'b_var: %.2f ' % b_var, )
                string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[episode] + ' Reward for tasks with priority 1: %5d' % ep_reward_task_prio_1[episode] + ' Reward for tasks with priority 2: %5d' % ep_reward_task_prio_2[episode] + ' Reward for tasks with priority 3: %5d' % ep_reward_task_prio_3[episode] + ' Reward for tasks with latency 1: %5d' % ep_reward_task_lat_1[episode] + ' Reward for tasks with latency 2: %5d' % ep_reward_task_lat_2[episode] + ' Reward for tasks with latency 3: %5d' % ep_reward_task_lat_3[episode] + ' Reward for tasks with application type 1: %5d' % ep_reward_app_typ_1[episode] + ' Reward for tasks with application type 2: %5d' % ep_reward_app_typ_2[episode] + ' Reward for tasks with application type 3: %5d' % ep_reward_app_typ_3[episode] + ' Reward for tasks with application type 4: %5d' % ep_reward_app_typ_4[episode] + ' Penalization: %5d' % ep_penalization[episode] + ' Penalization for tasks with priority 1: %5d' % ep_penalization_task_prio_1[episode] + ' Penalization for tasks with priority 2: %5d' % ep_penalization_task_prio_2[episode] + ' Penalization for tasks with priority 3: %5d' % ep_penalization_task_prio_3[episode] + ' Penalization for tasks with latency 1: %5d' % ep_penalization_task_lat_1[episode] + ' Penalization for tasks with latency 2: %5d' % ep_penalization_task_lat_2[episode] + ' Penalization for tasks with latency 3: %5d' % ep_penalization_task_lat_3[episode] + ' Penalization for tasks with application type 1: %5d' % ep_penalization_app_typ_1[episode] + ' Penalization for tasks with application type 2: %5d' % ep_penalization_app_typ_2[episode] + ' Penalization for tasks with application type 3: %5d' % ep_penalization_app_typ_3[episode] + ' Penalization for tasks with application type 4: %5d' % ep_penalization_app_typ_4[episode] + ' Usage history: ' + np.array2string(ep_usage_history[episode], precision=2, separator=',',
                      suppress_small=True) +' Security requirement: %5d' % ep_security_requirement[episode] + '###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var
                epoch_inf.append(string)
                '''
                # variation change
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1
                '''
                var_counter += 1

        # end the episode
        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        episode += 1

    # make directory
    dir_name = 'output/' +  str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '_' + ALGORITHM + '_' + METHOD + '_' + \
               CONCEPT + '_' +  LATENCY_REQUIREMENTS + '_' + str(r_dim) + 'u' + '_' +  str(int(o_dim / r_dim)) + 'e' + '_for ' + \
               str(MAX_EP_STEPS) + ' steps'
    if (os.path.isdir(dir_name)):
        os.rmdir(dir_name)
    os.makedirs(dir_name)
    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i+1 for i in range(episode)], ep_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')
    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')
    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')
    # mean of the rewards
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" + str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with priority 1 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_prio_1[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with priority 1:" + str(np.mean(ep_reward_task_prio_1[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with priority 2 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_prio_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with priority 2:" + str(np.mean(ep_reward_task_prio_2[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with priority 3 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_prio_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with priority 3:" + str(np.mean(ep_reward_task_prio_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with latency 1 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_lat_1[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with latency 1:" + str(np.mean(ep_reward_task_lat_1[-LEARNING_MAX_EPISODE:])) + '\n\n')   
    print("the mean of the rewards for tasks with latency 2 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_lat_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with latency 2:" + str(np.mean(ep_reward_task_lat_2[-LEARNING_MAX_EPISODE:])) + '\n\n')   
    print("the mean of the rewards for tasks with latency 3 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_task_lat_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with latency 3:" + str(np.mean(ep_reward_task_lat_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with application type 1 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_app_typ_1[-LEARNING_MAX_EPISODE:])))    
    f.write("the mean of the rewards for tasks with application type 1:" + str(np.mean(ep_reward_app_typ_1[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with application type 2 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_app_typ_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with application type 2:" + str(np.mean(ep_reward_app_typ_2[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with application type 3 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_app_typ_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with application type 3:" + str(np.mean(ep_reward_app_typ_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the rewards for tasks with application type 4 in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward_app_typ_4[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards for tasks with application type 4:" + str(np.mean(ep_reward_app_typ_4[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the security requirement in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_security_requirement[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the security requirement:" + str(np.mean(ep_security_requirement[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # mean of the penalizations
    print("the mean of the penalizations in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations:" + str(np.mean(ep_penalization[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with priority 1 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_prio_1[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with priority 1:" + str(np.mean(ep_penalization_task_prio_1[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with priority 2 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_prio_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with priority 2:" + str(np.mean(ep_penalization_task_prio_2[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with priority 3 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_prio_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with priority 3:" + str(np.mean(ep_penalization_task_prio_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with latency 1 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_lat_1[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with latency 1:" + str(np.mean(ep_penalization_task_lat_1[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with latency 2 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_lat_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with latency 2:" + str(np.mean(ep_penalization_task_lat_2[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with latency 3 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_task_lat_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with latency 3:" + str(np.mean(ep_penalization_task_lat_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with application type 1 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_app_typ_1[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with application type 1:" + str(np.mean(ep_penalization_app_typ_1[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with application type 2 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_app_typ_2[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with application type 2:" + str(np.mean(ep_penalization_app_typ_2[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with application type 3 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_app_typ_3[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with application type 3:" + str(np.mean(ep_penalization_app_typ_3[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the mean of the penalizations for tasks with application type 4 in the last", LEARNING_MAX_EPISODE, " epochs:",    str(np.mean(ep_penalization_app_typ_4[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the penalizations for tasks with application type 4:" + str(np.mean(ep_penalization_app_typ_4[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation
    print("the standard deviation of the rewards:", str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" + str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the rewards:", str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" + str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    print("the standard deviation of the penalizations:", str(np.std(ep_penalization[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the penalizations:" + str(np.std(ep_penalization[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the penalizations:", str(max(ep_penalization[-LEARNING_MAX_EPISODE:]) - min(ep_penalization[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the penalizations:" + str(max(ep_penalization[-LEARNING_MAX_EPISODE:]) - min(ep_penalization[-LEARNING_MAX_EPISODE:])) + '\n\n')
    std_array = []
    for i in range(LEARNING_MAX_EPISODE):
        print("the mean of the usage history in the epoch: ", i, " is: ", str(np.mean(ep_usage_history[-i:])))
        f.write("the mean of the usage history in the epoch: " + str(i) + " is: "+ str(np.mean(ep_usage_history[-i:])) + '\n\n')  
        print("the standard deviation of the usage history in the epoch: ", i,  " is: ", str(np.std(ep_usage_history[-i:])))
        f.write("the standard deviation of the usage history in the epoch: " + str(i) + " is: " + str(np.std(ep_usage_history[-i:])) + '\n\n')
        std_array.append(np.std(ep_usage_history[-i:]))
    print("the mean of the standard deviation of the usage history in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(std_array)))    
    f.write("the mean of the standard deviation of the usage history: " + str(np.mean(std_array)) + '\n\n')        
    f.close()

