"""
Test cases
"""
import action
from agent import Agent
from train import *
from util import *

###########################
###  Test for Sampling  ###
###########################

from sample import generate_examples

def io_generation_test1():
    generate_examples(1, True)

def io_generation_test2():
    inputs = ['john Smith',
              'DOUG Q. Macklin',
              'Frank Lee (123)',
              'Laura Jane Jones',
              'Steve P. Green (9)']
    actions = [action.GetToken1('Word'),
               action.GetToken2(-1),
               action.Commit(),
               action.ConstStr(','),
               action.Commit(),
               action.ConstStr(' '),
               action.Commit(),
               action.GetToken1('Word'),
               action.GetToken2(0),
               action.ToCase('Proper'),
               action.Commit()]
    outputs = ['' for _ in inputs]
    print(execute_actions(actions, action.RobState.new(inputs, outputs)))

def io_generation_test3():
    pstate = action.RobState.new(["12A", "2A4", "A45", "4&6", "&67"],
                                   ["", "", "", "", ""])
    print (pstate)
    fs = [action.ToCase("Lower"),
          action.Replace1("&"),
          action.Replace2("["),
          action.Substr1(1),
          action.Substr2(2),
          action.Commit()
         ]

    print(execute_actions(fs, pstate))

def io_generation_test4():
    pstate = action.RobState.new(['Mr.Pu', 'Mr.Poo'],
                                   ['', ''])

    fs = [action.GetToken1('Word'),
          action.GetToken2(1),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetUpTo('.'),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetFrom('.'),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetFirst1('Word'),
          action.GetFirst2(2),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetAll('Word'),
          action.Commit()]
    print(execute_actions(fs, pstate))

def io_generation_test5():
    pstate = action.RobState.new(["(hello)1)23", "(mis)ter)123"],
                                   ["HELLO", "MIS"])
    fs = [action.GetSpan1("("),
          action.GetSpan2(0),
          action.GetSpan3("End"),
          action.GetSpan4(")"),
          action.GetSpan5(0),
          action.GetSpan6("Start"),
          action.ToCase("AllCaps"),
          action.Commit()]

    print(execute_actions(fs, pstate))

def io_generation_test6():
    print(get_supervised_sample())

def train_test0():
    states, actions = get_supervised_sample()
    print("get states:", len(states))
    print("get actions:", len(actions))
    agent = Agent(action.ALL_ACTIONS)
    for i in range(400):
        loss = agent.learn_supervised(states, actions)
        if i%10 == 0: print(f"iteration {i}, loss: {loss.item()}")
    pred_actions = agent.best_actions(states)
    print("real actions:")
    print(actions)
    print("model actions:")
    print(pred_actions)

if __name__ == '__main__':
    # io_generation_test1()
    io_generation_test2()
    io_generation_test3()
    io_generation_test4()
    io_generation_test5()
    io_generation_test6()
    train_test0()
