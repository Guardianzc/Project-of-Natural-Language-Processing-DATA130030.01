import argparse
from Trainer import Solver

def main(config):
    solver = Solver(config)

    if config.mode == 'train':
        print("Start modeling")
        solver.modeling()
        print("Start testing")
        solver.test(config)
    
    elif config.mode == 'test':
        print("Load model from: {}".format(config.model_path))
        #solver.test(config)
        solver.test(config)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #mode
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--maxchangecount', type=int, default=917)
    #save
    parser.add_argument('--model_path', type=str, default='./elements/')
    parser.add_argument('--data_root', type=str, default='./elements/')

    # Hyper-parameter
    parser.add_argument('--weight', type=str, default='0.4 0.0 0.6')
    parser.add_argument('--smoothingMethod', type=str, default='add-k smoothing') #'add-k smoothing \ Absolute Discounting Interpolation'
    parser.add_argument('--k_in_addk', type=int, default=1)
    
    
    config = parser.parse_args() #return a name space
    main(config)

    

