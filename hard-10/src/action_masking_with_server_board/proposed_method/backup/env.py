import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from render import Demo

#****************************PMC-start***************************
import random
import copy
#****************************PMC-end***************************

#####################  hyper parameters  ####################
LOCATION = "KAIST"
USER_NUM = 25 #10
EDGE_NUM = 10
LIMIT = 4
MAX_EP_STEPS = 1000 #3000
TXT_NUM = 92
r_bound = 1e9 * 0.063
b_bound = 1e9

#####################  function  ####################
def trans_rate(user_loc, edge_loc):
    B = 2e6
    P = 0.25
    d = np.sqrt(np.sum(np.square(user_loc[0] - edge_loc))) + 0.01
    h = 4.11 * math.pow(3e8 / (4 * math.pi * 915e6 * d), 2)
    N = 1e-10
    return B * math.log2(1 + P * h / N)

def BandwidthTable(edge_num):
    BandwidthTable = np.zeros((edge_num, edge_num))
    for i in range(0, edge_num):
        for j in range(i+1, edge_num):
                BandwidthTable[i][j] = 1e9
    return BandwidthTable

def two_to_one(two_table):
    one_table = two_table.flatten()
    return one_table

def generate_state(two_table, U, E, usage_history, x_min, y_min):
    # initial
    one_table = two_to_one(two_table)
    S = np.zeros((len(E) + one_table.size + len(U) + len(U)*2))
    # transform
    count = 0
    # available resource of each edge server
    for edge in E:
        S[count] = edge.capability/(r_bound*10)
        count += 1
    # available bandwidth of each connection
    for i in range(len(one_table)):
        S[count] = one_table[i]/(b_bound*10)
        count += 1
    # offloading of each user
    for user in U:
        S[count] = user.req.edge_id/100
        count += 1
    # location of the user
    for user in U:
        S[count] = (user.loc[0][0] + abs(x_min))/1e5
        S[count+1] = (user.loc[0][1] + abs(y_min))/1e5
        count += 2   
        
#****************************PMC-start-state-transformation***************************    
    #print("state before transformation: ", S)    
    #S = transform_state(S, U, E, usage_history, "user_group")
    #S = transform_state(S, U, E, usage_history, "user_card_number")
    #S = transform_state(S, U, E, usage_history, "user_device_type")
    #S = transform_state(S, U, E, usage_history, "user_device_OS")
    #S = transform_state(S, U, E, usage_history, "usage_history")
    #S = transform_state(S, U, E, usage_history, "server_group")
    #S = transform_state(S, U, E, usage_history, "server_board")
    S = transform_state(S, U, E, usage_history, "server_workload")
    S = transform_state(S, U, E, usage_history, "server_limit")
    #S = transform_state(S, U, E, usage_history, "server_cost")
    #S = transform_state(S, U, E, usage_history, "application_type")
    #S = transform_state(S, U, E, usage_history, "task_latency")
    #S = transform_state(S, U, E, usage_history, "task_priority")
    #print("state after transformation: ", S)
#****************************PMC-end-state-transformation***************************  
    return S
    
#****************************PMC-start-state-transformation***************************
def transform_state(S, U, E, usage_history, concept): 
    #print("size of state-before transformation: ", S.size)
    count = S.size
    
    if concept == "user_group":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.user_group
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "user_card_number":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.user_card_number
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "user_device_type":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.user_device_type
            count += 1
        #print("size of state-after transformation: ", S.size)

    if concept == "user_device_OS":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.user_device_OS
            count += 1
        #print("size of state-after transformation: ", S.size)

    if concept == "usage_history":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] =  usage_history[user.user_id]
            count += 1
        #print("size of state-after transformation: ", S.size)
                
    if concept == "server_group":
        S = np.pad(S, (0,  len(E)), 'constant')
        for edge in E:
            S[count] = edge.server_group
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "server_board":
        S = np.pad(S, (0,  len(E)), 'constant')
        for edge in E:
            S[count] = edge.server_board
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "server_workload":
        S = np.pad(S, (0,  len(E)), 'constant')
        for edge in E:
            S[count] = len(edge.server_workload)
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "server_limit":
        S = np.pad(S, (0,  len(E)), 'constant')
        for edge in E:
            S[count] = edge.limit
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "server_cost":
        S = np.pad(S, (0,  len(E)), 'constant')
        for edge in E:
            S[count] = edge.server_cost
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "application_type":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.req.tasktype.application_type
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "task_latency":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.req.tasktype.task_latency
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    if concept == "task_priority":
        S = np.pad(S, (0,  len(U)), 'constant')
        for user in U:
            S[count] = user.req.tasktype.task_priority
            count += 1
        #print("size of state-after transformation: ", S.size)
        
    return S
#****************************PMC-end-state-transformation***************************
   
def generate_action(R, B, O):
    # resource
    a = np.zeros(USER_NUM + USER_NUM + EDGE_NUM * USER_NUM)
    a[:USER_NUM] = R / r_bound
    # bandwidth
    a[USER_NUM:USER_NUM + USER_NUM] = B / b_bound
    # offload
    base = USER_NUM + USER_NUM
    for user_id in range(USER_NUM):
        a[base + int(O[user_id])] = 1
        base += EDGE_NUM
    return a

def get_minimum():
    cal = np.zeros((1, 2))
    for data_num in range(TXT_NUM):
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_name + ".txt"
        file_path = "data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num
        line_num = 0
        for line in f1:
            line_num += 1
        # collect the data from the .txt
        data = np.zeros((line_num, 2))
        index = 0
        for line in f1:
            data[index][0] = line.split()[1]  # x
            data[index][1] = line.split()[2]  # y
            index += 1
        # put data into the cal
        cal = np.vstack((cal, data))
    return min(cal[:, 0]), min(cal[:, 1])

def proper_edge_loc(edge_num):

    # initial the e_l
    e_l = np.zeros((edge_num, 2))
    # calculate the mean of the data
    group_num = math.floor(TXT_NUM / edge_num)
    edge_id = 0
    for base in range(0, group_num*edge_num, group_num):
        for data_num in range(base, base + group_num):
            data_name = str("%03d" % (data_num + 1))  # plus zero
            file_name = LOCATION + "_30sec_" + data_name + ".txt"
            file_path = "data/" + LOCATION + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            # get line_num and initial data
            line_num = 0
            for line in f1:
                line_num += 1
            data = np.zeros((line_num, 2))
            # collect the data from the .txt
            index = 0
            for line in f1:
                data[index][0] = line.split()[1]  # x
                data[index][1] = line.split()[2]  # y
                index += 1
            # stack the collected data
            if data_num % group_num == 0:
                cal = data
            else:
                cal = np.vstack((cal, data))
        e_l[edge_id] = np.mean(cal, axis=0)
        edge_id += 1
    return e_l

#############################UE###########################
class UE():
    def __init__(self, user_id, data_num):
    
#****************************PMC-start-adding-user_group***************************
        
        self.user_group = np.random.choice(np.arange(1, 4), p=[0.3, 0.3, 0.4])
        
        #print(user_id, " ", self.user_group) 
        
        '''
	0   2
	1   3
	2   1
	3   2
	4   3
	5   3
	6   3
	7   3
	8   2
	9   1
	
	3 -> 5
	2 -> 3
	1 -> 2
        '''
#****************************PMC-end-adding-user_group***************************

#****************************PMC-start-adding-user_card_number*******************

        generator = random.Random()
        generator.seed()        # Seed from current time        
        
        self.user_card_number = credit_card_number(generator, mastercardPrefixList, 16, 1)
        
        #print(user_id, " ", self.user_card_number) 
        
        '''
	0   5441189301408470.0
	1   5397709410016770.0
	2   5497971715232440.0
	3   5375247678243680.0
	4   5476678473475530.0
	5   5597598108810370.0
	6   5233807811025120.0
	7   5131504085133750.0
	8   5493875548638190.0
	9   5153358991245610.0

        '''
#****************************PMC-end-adding-user_card_number***********************

#****************************PMC-start-adding-user_device_type*******************

        self.user_device_type = random. randint(1,3) # 1 for smartphones, 2 for wearable gadgets, and 3 for laptops
                
        #print(user_id, " ", self.user_device_type) 
        
        '''
	0   1
	1   2
	2   2
	3   2
	4   1
	5   2
	6   2
	7   1
	8   1
	9   3
        '''
#****************************PMC-end-adding-user_device_type***********************

#****************************PMC-start-adding-user_device_OS*******************

        self.user_device_OS = random. randint(1,4) # 1 for Windows, 2 for Linux, 3 for Android, and 4 for iOS
                
        #print(user_id, " ", self.user_device_OS) 
        
        '''
	0   1
	1   2
	2   4
	3   2
	4   2
	5   2
	6   1
	7   1
	8   4
	9   2
        '''
#****************************PMC-end-adding-user_device_OS***********************

        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_num + ".txt"
        file_path = "data/" + LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))

        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30
        self.loc[0] = self.mob[0]

    def generate_request(self, edge_id):
        self.req = Request(self.user_id, edge_id)

    def request_update(self):
        # default request.state == 5 means disconnection ,6 means migration
        if self.req.state == 5:
            self.req.timer += 1
        else:
            self.req.timer = 0
            if self.req.state == 0:
                self.req.state = 1
                self.req.u2e_size = self.req.tasktype.req_u2e_size
                self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
            elif self.req.state == 1:
                if self.req.u2e_size > 0:
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
                else:
                    self.req.state = 2
                    self.req.process_size = self.req.tasktype.process_loading
                    self.req.process_size -= self.req.resource
            elif self.req.state == 2:
                if self.req.process_size > 0:
                    self.req.process_size -= self.req.resource
                else:
                    self.req.state = 3
                    self.req.e2u_size = self.req.tasktype.req_e2u_size
                    self.req.e2u_size -= 10000  # value is small,so simplify
            else:
                if self.req.e2u_size > 0:
                    self.req.e2u_size -= 10000  # B*math.log(1+SINR(self.user.loc, self.offloading_serv.loc), 2)/(8*time_scale)
                else:
                    self.req.state = 4

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]   # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf

class Request():
    def __init__(self, user_id, edge_id):
        # id
        self.user_id = user_id
        self.edge_id = edge_id
        self.edge_loc = 0
        # state
        self.state = 5     # 5: not connect
        self.pre_state=5
        # transmission size
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge state
        self.resource = 0
        self.mig_size = 0
        # tasktype
        self.tasktype = TaskType()
        self.last_offlaoding = 0
        # timer
        self.timer = 0

class TaskType():
    def __init__(self):
        ##Objection detection: VOC SSD300
        # transmission
        self.req_u2e_size = 300 * 300 * 3 * 1
        self.process_loading = 300 * 300 * 3 * 4
        self.req_e2u_size = 4 * 4 + 20 * 4
        
#****************************PMC-start-adding-application_type*******************

        #Simple scenario
        self.application_type = np.random.choice(np.arange(1, 5), p=[0.05, 0.15, 0.55, 0.25]) #1 is remote healthcare, 2 is VoIP, 3 is data collection, and 4 is entertainment.
        
        #Medium scenario
        #self.application_type = np.random.choice(np.arange(1, 5), p=[0.1, 0.30, 0.35, 0.25]) #1 is remote healthcare, 2 is VoIP, 3 is data collection, and 4 is entertainment.
        
        #Hard scenario
        #self.application_type = np.random.choice(np.arange(1, 5), p=[0.20, 0.40, 0.10, 0.30]) #1 is remote healthcare, 2 is VoIP, 3 is data collection, and 4 is entertainment.       
        
        #print("application_type: ", self.application_type) 
        
        '''
	application_type:  3
	application_type:  3
	application_type:  2
	application_type:  2
	application_type:  2
	application_type:  2
	application_type:  1
	application_type:  4
	application_type:  3
	application_type:  4
	application_type:  3
	application_type:  3
	application_type:  3
	application_type:  3
	application_type:  4
        '''

#****************************PMC-end-adding-application_type*******************

#****************************PMC-start-adding-task_latency*******************

        if self.application_type == 1: self.task_latency = 1 #remote health care is a very low latency task.
        if self.application_type == 2: self.task_latency = 1 #VoIP is a very low latency task.        
        if self.application_type == 3: self.task_latency = 3 #data collection is a high latency task.
        if self.application_type == 4: self.task_latency = 2 #entertainment is a low latency task.
                        
        #print("task latency: ", self.task_latency) 
        
        '''
	task latency:  1
	task latency:  3
	task latency:  2
	task latency:  3
	task latency:  3
	task latency:  3
	task latency:  3
	task latency:  2
	task latency:  3
	task latency:  3
	task latency:  3
	task latency:  3
	task latency:  3
	task latency:  2
	task latency:  3
	task latency:  2
	task latency:  2
        '''

#****************************PMC-end-adding-task_latency*******************

#****************************PMC-start-adding-task_priority*******************

        if self.application_type == 1: self.task_priority = 3 #remote health care with high priority.
        if self.application_type == 2: self.task_priority = 2 #VoIP with middle priority.        
        if self.application_type == 3: self.task_priority = 1 #data collection with low priority.
        if self.application_type == 4: self.task_priority = 1 #entertainment with low priority.
                        
        #print("task priority: ", self.task_priority) 
        
        '''
	task priority:  1
	task priority:  1
	task priority:  1
	task priority:  1
	task priority:  3
	task priority:  3
	task priority:  1
	task priority:  1
	task priority:  1
	task priority:  2
	task priority:  1
	task priority:  3
	task priority:  1
	task priority:  1
	task priority:  1
        '''

#****************************PMC-end-adding-task_priority*******************
        
        # migration
        self.migration_size = 2e9
    def task_inf(self):
        return "req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)

#############################EdgeServer###################

class EdgeServer():
    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number
        self.loc = loc
        self.capability = 1e9 * 0.063
        self.server_workload = []        
        #self.limit = LIMIT
        
#****************************PMC-start-adding-server_limit***************************

        self.limit = np.random.choice(np.arange(1, 5), p=[0.2, 0.3, 0.3, 0.2])
        #print(edge_id, " ", self.limit) 

        '''
	0   4
	1   3
	2   4
	3   2
	4   3
	5   3
	6   1
	7   1
	8   4
	9   3
	
	4 -> 3
	3 -> 4
	2 -> 1
	1 -> 2
        '''        
#****************************PMC-end-adding-server_limit***************************
      
        self.connection_num = 0
        
#****************************PMC-start-adding-server_group***************************
        self.server_group = np.random.choice(np.arange(1, 4), p=[0.3, 0.3, 0.4])
        
        #print(edge_id, " ", self.server_group) 
        
        '''
	0   1
	1   2
	2   3
	3   2
	4   3
	5   2
	6   2
	7   2
	8   3
	9   2
	
	3 -> 3
	2 -> 6
	1 -> 1
        '''
#****************************PMC-end-adding-server_group***************************

#****************************PMC-start-adding-server_board***************************
        self.server_board = np.random.choice(np.arange(0, 2), p=[0.3, 0.7]) # 0 is equal to the low server board (lower than or equal to 800) and 1 is equal to the high server board (higher than 800)
        
        #print(edge_id, " ", self.server_board) 
        
        '''
	0   1
	1   0
	2   1
	3   1
	4   0
	5   1
	6   0
	7   1
	8   1
	9   1

	0 -> 3
	1-> 7
        '''
#****************************PMC-end-adding-server_board***************************

#****************************PMC-start-adding-server_cost***************************
        self.server_cost = np.random.choice(np.arange(0, 2), p=[0.8, 0.2]) # 0 is equal to the free server and 1 is equal to the paid serverj, 
        
        #print(edge_id, " ", self.server_cost) 
        
        '''
	0   1
	1   0
	2   0
	3   0
	4   0
	5   0
	6   1
	7   0
	8   0
	9   0

	0 -> 8
	1-> 2
        '''
#****************************PMC-end-adding-server_cost***************************

    def maintain_request(self, R, U):
        for user in U:
            # the number of the connection user
            self.connection_num = 0
            for user_id in self.server_workload:
                if U[user_id].req.state != 6:
                    self.connection_num += 1
            # maintain the request
            if user.req.edge_id == self.edge_id and self.capability - R[user.user_id] > 0:
                # maintain the preliminary connection
                if user.req.user_id not in self.server_workload and self.connection_num+1 <= self.limit:
                    # first time : do not belong to any edge(server_workload)
                    self.server_workload.append(user.user_id)  # add to the server_workload
                    user.req.state = 0  # prepare to connect
                    # notify the request
                    user.req.edge_id = self.edge_id
                    user.req.edge_loc = self.loc

                # dispatch the resource
                user.req.resource = R[user.user_id]
                self.capability -= R[user.user_id]

    def migration_update(self, O, B, table, U, E):

        # maintain the the migration
        for user_id in self.server_workload:
            # prepare to migration
            if U[user_id].req.edge_id != O[user_id]:
                # initial
                ini_edge = int(U[user_id].req.edge_id)
                target_edge = int(O[user_id])
                if table[ini_edge][target_edge] - B[user_id] >= 0:
                    # on the way to migration, but offloading to another edge computer(step 1)
                    if U[user_id].req.state == 6 and target_edge != U[user_id].req.last_offlaoding:
                        # reduce the bandwidth
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        #print("user", U[user_id].req.user_id, ":migration step 1")
                    # first try to migration(step 1)
                    elif U[user_id].req.state != 6:
                        table[ini_edge][target_edge] -= B[user_id]
                        # start migration
                        U[user_id].req.mig_size = U[user_id].req.tasktype.migration_size
                        U[user_id].req.mig_size -= B[user_id]
                        # store the pre state
                        U[user_id].req.pre_state = U[user_id].req.state
                        # on the way to migration, disconnect to the old edge
                        U[user_id].req.state = 6
                        #print("user", U[user_id].req.user_id, ":migration step 1")
                    elif U[user_id].req.state == 6 and target_edge == U[user_id].req.last_offlaoding:
                        # keep migration(step 2)
                        if U[user_id].req.mig_size > 0:
                            # reduce the bandwidth
                            table[ini_edge][target_edge] -= B[user_id]
                            U[user_id].req.mig_size -= B[user_id]
                            #print("user", U[user_id].req.user_id, ":migration step 2")
                        # end the migration(step 3)
                        else:
                            # the number of the connection user
                            target_connection_num = 0
                            for target_user_id in E[target_edge].server_workload:
                                if U[target_user_id].req.state != 6:
                                    target_connection_num += 1
                            #print("user", U[user_id].req.user_id, ":migration step 3")
                            # change to another edge
                            if E[target_edge].capability - U[user_id].req.resource >= 0 and target_connection_num + 1 <= E[target_edge].limit:
                                # register in the new edge
                                E[target_edge].capability -= U[user_id].req.resource
                                E[target_edge].server_workload.append(user_id)
                                self.server_workload.remove(user_id)
                                # update the request
                                # id
                                U[user_id].req.edge_id = E[target_edge].edge_id
                                U[user_id].req.edge_loc = E[target_edge].loc
                                # release the pre-state, continue to transmission process
                                U[user_id].req.state = U[user_id].req.pre_state
                                #print("user", U[user_id].req.user_id, ":migration finish")
            #store pre_offloading
            U[user_id].req.last_offlaoding = int(O[user_id])

        return table

    #release the all resource
    def release(self):
        self.capability = 1e9 * 0.063

#############################Policy#######################

class priority_policy():
    def generate_priority(self, U, E, priority):
        for user in U:
            # get a list of the offloading priority
            dist = np.zeros(EDGE_NUM)
            for edge in E:
                dist[edge.edge_id] = np.sqrt(np.sum(np.square(user.loc[0] - edge.loc)))
            dist_sort = np.sort(dist)
            for index in range(EDGE_NUM):
                priority[user.user_id][index] = np.argwhere(dist == dist_sort[index])[0]
        return priority

    def indicate_edge(self, O, U, priority):
        edge_limit = np.ones((EDGE_NUM)) * LIMIT
        for user in U:
            for index in range(EDGE_NUM):
                if edge_limit[int(priority[user.user_id][index])] - 1 >= 0:
                    edge_limit[int(priority[user.user_id][index])] -= 1
                    O[user.user_id] = priority[user.user_id][index]
                    break
        return O

    def resource_update(self, R, E ,U):
        for edge in E:
            # count the number of the connection user
            connect_num = 0
            for user_id in edge.server_workload:
                if U[user_id].req.state != 5 and U[user_id].req.state != 6:
                    connect_num += 1
            # dispatch the resource to the connection user
            for user_id in edge.server_workload:
                # no need to provide resource to the disconnecting users
                if U[user_id].req.state == 5 or U[user_id].req.state == 6:
                    R[user_id] = 0
                # provide resource to connecting users
                else:
                    R[user_id] = edge.capability/(connect_num+2)  # reserve the resource to those want to migration
        return R

    def bandwidth_update(self, O, table, B, U, E):
        for user in U:
            share_number = 1
            ini_edge = int(user.req.edge_id)
            target_edge = int(O[user.req.user_id])
            # no need to migrate
            if ini_edge == target_edge:
                B[user.req.user_id] = 0
            # provide bandwidth to migrate
            else:
                # share bandwidth with user from migration edge
                for user_id in E[target_edge].server_workload:
                    if O[user_id] == ini_edge:
                        share_number += 1
                # share bandwidth with the user from the original edge to migration edge
                for ini_user_id in E[ini_edge].server_workload:
                    if ini_user_id != user.req.user_id and O[ini_user_id] == target_edge:
                        share_number += 1
                # allocate the bandwidth
                B[user.req.user_id] = table[min(ini_edge, target_edge)][max(ini_edge, target_edge)] / (share_number+2)

        return B

#############################Env###########################

class Env():
    def __init__(self):
        self.step = 30
        self.time = 0
        self.edge_num = EDGE_NUM  # the number of servers
        self.user_num = USER_NUM  # the number of users
        # define environment object
        self.reward_all = []
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        self.rewards = 0
        self.R = np.zeros((self.user_num))
        self.O = np.zeros((self.user_num))
        self.B = np.zeros((self.user_num))
        self.table = BandwidthTable(self.edge_num)
        self.priority = np.zeros((self.user_num, self.edge_num))
        self. E = []
        self.x_min, self.y_min = get_minimum()

        self.e_l = 0
        self.model = 0

    def get_inf(self):
        # s_dim
        self.reset()
        s = generate_state(self.table, self.U, self.E, self.usage_history, self.x_min, self.y_min)        
        s_dim = s.size

        # a_dim
        r_dim = len(self.U)
        b_dim = len(self.U)
        o_dim = self.edge_num * len(self.U)

        # maximum resource
        r_bound = self.E[0].capability

        # maximum bandwidth
        b_bound = self.table[0][1]
        b_bound = b_bound.astype(np.float32)

        # task size
        task = TaskType()
        task_inf = task.task_inf()

        return s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, LIMIT, LOCATION

    def reset(self):
        # reset time
        self.time = 0
        # reward
        self.reward_all = []
        # user
        self.U = []
        self.fin_req_count = 0
        self.prev_count = 0
        data_num = random.sample(list(range(TXT_NUM)), self.user_num)
        for i in range(self.user_num):
            new_user = UE(i, data_num[i])
            self.U.append(new_user)
        # Resource
        self.R = np.zeros((self.user_num))
        # Offlaoding
        self.O = np.zeros((self.user_num))
        # bandwidth
        self.B = np.zeros((self.user_num))
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # server
        self.E = []

#****************************PMC-start-adding-usage_history***************************  
      
        self.usage_history = np.zeros((self.user_num))
        
#****************************PMC-end-adding-usage_history***************************
        
        e_l = proper_edge_loc(self.edge_num)
        for i in range(self.edge_num):
            new_e = EdgeServer(i, e_l[i, :])
            self.E.append(new_e)
            """
            print("edge", new_e.edge_id, "'s loc:\n", new_e.loc)
        print("========================================================")
        """
        # model
        self.model = priority_policy()

        # initialize the request
        self.priority = self.model.generate_priority(self.U, self.E, self.priority)
        self.O = self.model.indicate_edge(self.O, self.U, self.priority)
        for user in self.U:
            user.generate_request(self.O[user.user_id])
        return generate_state(self.table, self.U, self.E, self.usage_history, self.x_min, self.y_min)

    def ddpg_step_forward(self, a, r_dim, b_dim):
        # release the bandwidth
        self.table = BandwidthTable(self.edge_num)
        # release the resource
        for edge in self.E:
            edge.release()

        # update the policy every second
        # resource update
        self.R = a[:r_dim]
        # bandwidth update
        self.B = a[r_dim:r_dim + b_dim]
        # offloading update
        base = r_dim + b_dim
        
        s_ = generate_state(self.table, self.U, self.E, self.usage_history, self.x_min, self.y_min)        
        #print("start of selecting action")
        for user_id in range(self.user_num):
            prob_weights = a[base:base + self.edge_num]   
            
                            
#****************************PMC-start-action_masking***************************

            prob_weights = action_masking(s_, prob_weights, "server_limit")
                
#****************************PMC-end-action_masking***************************
        
            action = np.random.choice(range(len(prob_weights)), p=prob_weights.ravel())  # select action w.r.t the actions prob
            base += self.edge_num     
                                  
            self.O[user_id] = action
            
            #print(user_id, action)
        #print("end of selecting action")   
        task_without_delay = 0
        task_with_delay = 0                   
        # request update
        for user in self.U:
            # update the state of the request
            user.request_update()
            if user.req.timer == 0: task_without_delay += 1
            if user.req.timer > 0: task_with_delay += 1
            if user.req.timer >= 25: #5
                user.generate_request(self.O[user.user_id])  # offload according to the priority
            # it has already finished the request
            if user.req.state == 4:
                # rewards
                self.fin_req_count += 1
                  
#****************************PMC-start-adding-usage_history***************************

                self.usage_history[user.user_id] += 1
                #print("user id: ", user.user_id, "usage history: ", self.usage_history[user.user_id])
                
#****************************PMC-end-adding-usage_history***************************

                user.req.state = 5  # request turn to "disconnect"
                self.E[int(user.req.edge_id)].server_workload.remove(user.req.user_id)
                user.generate_request(self.O[user.user_id])  # offload according to the priority

        print("in the proposed algorithm: ")
        #print("task_without_delay: ", task_without_delay) 
        #print("task_with_delay: ", task_with_delay)  
        print("algorithm's performance: ", task_without_delay - task_with_delay)        
        # edge update
        for edge in self.E:
            edge.maintain_request(self.R, self.U)
            self.table = edge.migration_update(self.O, self.B, self.table, self.U, self.E)

        # rewards
        self.rewards = self.fin_req_count - self.prev_count
        self.prev_count = self.fin_req_count

        # every user start to move
        if self.time % self.step == 0:
            for user in self.U:
                user.mobility_update(self.time)

        # update time
        self.time += 1

        # return s_, r
        return generate_state(self.table, self.U, self.E, self.usage_history, self.x_min, self.y_min), self.rewards

    def text_render(self):
        print("R:", self.R)
        print("B:", self.B)
        """
        base = USER_NUM +USER_NUM
        for user in range(len(self.U)):
            print("user", user, " offload probabilty:", a[base:base + self.edge_num])
            base += self.edge_num
        """
        print("O:", self.O)
        for user in self.U:
            print("user", user.user_id, "'s loc:\n", user.loc)
            print("request state:", user.req.state)
            print("edge serve:", user.req.edge_id)
        for edge in self.E:
            print("edge", edge.edge_id, "server_workload:", edge.server_workload)
        print("reward:", self.rewards)
        print("=====================update==============================")

    def initial_screen_demo(self):
        self.canvas = Demo(self.E, self.U, self.O, MAX_EP_STEPS)

    def screen_demo(self):
        self.canvas.draw(self.E, self.U, self.O)

#****************************PMC-start-action_masking***************************
        
def action_masking(s_, prob_weights, concept):
    #print("observation: ", s_)
    #print("size of observation: ", s_.size)
    if (concept == "server_limit"):
        #starting index of server_workload in s_: 140
        #starting index of server_limit in s_: 150
        number_of_full_servers = 0
        full_servers = []
        sum_remain_server_limit = 0
        for i in range (10):
             if s_[140+i] == s_[150+i]:
                 #print("server ", i, "is full")                 
                 '''
                 3.         0.         0.         2.
                 0.         1.         0.         0.         2.         0.
		3.         3.         3.         2.         4.         4.
 		2.         2.         2.         4.        ]
		server  0 is full
		server  3 is full
		server  8 is full
                 '''
                 number_of_full_servers += 1
                 full_servers.append(i)
             sum_remain_server_limit += (s_[150+i] - s_[140+i])
                 
        #print("sum_remain_server_limit: ", sum_remain_server_limit)
                 
        #print("number_of_full_servers: ", number_of_full_servers)
        #print("full_servers: ", full_servers)   
              
        number_of_free_servers = 10 -  number_of_full_servers
        #print("number_of_free_servers: ", number_of_free_servers)  
              
        for i in range (10):
             if s_[140+i] == s_[150+i]:
                 for j in range (10):
                     if j not in full_servers:
                         #prob_weights[j] = prob_weights[j] + (prob_weights[i]/number_of_free_servers) #first version
                         #print("prob_weights[", i, "]: ", prob_weights[i])
                         #print("prob_weights[",j,"] before increasing: ", prob_weights[j])
                         if sum_remain_server_limit != 0: 
                             prob_weights[j] = prob_weights[j] + prob_weights[i] * ((s_[150+j] - s_[140+j])/sum_remain_server_limit) #second version
                             #print("rate of increasing: ", s_[150+j] - s_[140+j])
                             #print("increasing amount: ", prob_weights[i] * ((s_[150+j] - s_[140+j])/sum_remain_server_limit))
                             #print("prob_weights[", j, "] after increasing: ", prob_weights[j])
                 prob_weights[i] = 0

    return prob_weights 
            
#****************************PMC-end-action_masking***************************  
        
#****************************PMC-start-credit-card-numbers-generator***************************
#reference: https://github.com/eye9poob/python

visaPrefixList = [
        ['4', '5', '3', '9'],
        ['4', '5', '5', '6'],
        ['4', '9', '1', '6'],
        ['4', '5', '3', '2'],
        ['4', '9', '2', '9'],
        ['4', '0', '2', '4', '0', '0', '7', '1'],
        ['4', '4', '8', '6'],
        ['4', '7', '1', '6'],
        ['4']]

mastercardPrefixList = [
        ['5', '1'], ['5', '2'], ['5', '3'], ['5', '4'], ['5', '5']]

amexPrefixList = [['3', '4'], ['3', '7']]

discoverPrefixList = [['6', '0', '1', '1']]

dinersPrefixList = [
        ['3', '0', '0'],
        ['3', '0', '1'],
        ['3', '0', '2'],
        ['3', '0', '3'],
        ['3', '6'],
        ['3', '8']]

enRoutePrefixList = [['2', '0', '1', '4'], ['2', '1', '4', '9']]

jcbPrefixList = [['3', '5']]

voyagerPrefixList = [['8', '6', '9', '9']]

def completed_number(generator, prefix, length):
    """
    'prefix' is the start of the CC number as a string, any number of digits.
    'length' is the length of the CC number to generate. Typically 13 or 16
    """

    ccnumber = prefix

    # generate digits

    while len(ccnumber) < (length - 1):
        digit = str(generator.choice(range(0, 10)))
        ccnumber.append(digit)

    # Calculate sum

    sum = 0
    pos = 0

    reversedCCnumber = []
    reversedCCnumber.extend(ccnumber)
    reversedCCnumber.reverse()

    while pos < length - 1:

        odd = int(reversedCCnumber[pos]) * 2
        if odd > 9:
            odd -= 9

        sum += odd

        if pos != (length - 2):

            sum += int(reversedCCnumber[pos + 1])

        pos += 2

    # Calculate check digit

    checkdigit = ((sum / 10 + 1) * 10 - sum) % 10

    ccnumber.append(str(checkdigit))

    return ''.join(ccnumber)


def credit_card_number(rnd, prefixList, length, howMany):

    ccnumber = copy.copy(rnd.choice(prefixList))
    result = completed_number(rnd, ccnumber, length)

    '''
    result = []

    while len(result) < howMany:

        ccnumber = copy.copy(rnd.choice(prefixList))
        result.append(completed_number(rnd, ccnumber, length))
    '''
    return result

def output(title, numbers):

    result = []
    result.append(title)
    result.append('-' * len(title))
    result.append('\n'.join(numbers))
    result.append('')

    return '\n'.join(result)

#****************************PMC-end-credit-card-numbers-generator***************************
