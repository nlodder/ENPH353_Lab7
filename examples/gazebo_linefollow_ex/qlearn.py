import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon      # exploration constant
        self.alpha = alpha          # discount constant
        self.gamma = gamma          # discount factor
        self.actions = actions      # list of possible actions
        self.exploration = dict()   # dictionary to track random vs highest Q action choices for debugging purposes

    def loadQ(self, filename):
        '''
            @brief Load the Q state-action values from a pickle file.
        '''
        with open(filename + ".pickle", 'rb') as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
            @brief Save the Q state-action values in a pickle file.
        '''
        with open(filename + ".pickle", 'wb') as f:
            pickle.dump(self.q, f)
        
        # Save to CSV for easier human readability
        # CSV file should have 3 columns: state, action, Q value
        with open(filename + ".csv", 'w') as f:
            for key, value in self.q.items():
                f.write("{},{},{}\n".format(key[0], key[1], value))

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
            @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
            @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time.
            If there are multiple actions with the same max Q value, randomly chooses between them.
            @returns action or (action, q) if return_q is True
        '''
        if random.random() < self.epsilon: # random.random() returns a random float in [0.0, 1.0)
            action = random.choice(self.actions)
            self.exploration['random'] = self.exploration.get('random', 0) + 1
            if return_q:
                return action, self.getQ(state, action)
            else:
                return action
        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(max_actions) # break ties randomly
            self.exploration['max_q'] = self.exploration.get('max_q', 0) + 1
            if return_q:
                return action, max_q
            else:
                return action

    def learn(self, state1, action1, reward, state2):
        '''
            @brief updates the Q(state,value) dictionary using the bellman update equation.
            If action has not been taken before in this state then the Q value is 
            initialized to 0.0 before the update.
        '''
        current_Q = self.getQ(state1, action1)
        # if (state1, action1) not in self.q, we initialize it to 0 to prevent rewarding lucky results
        if current_Q == 0.0:
            current_Q = 0
        
        max_Q_next = max([self.getQ(state2, a) for a in self.actions])
        # Bellman update
        new_q = current_Q + self.alpha * (reward + self.gamma * max_Q_next - current_Q)

        self.q[(state1,action1)] = new_q
