from QRec import QRec
from util.config import ModelConf

if __name__ == '__main__':

    print('=' * 80)
    print('CLRec: An effective python-based recommendation model library.')
    print('=' * 80)
    print('Self-Supervised Recommenders:')
    print('ss1. SGL ss2. SimGCL')
    print('=' * 80)
    num = input('please enter the number of the model you want to run:')
    import time

    s = time.time()
    # Register your model here and add the conf file into the config directory
    models = {'ss1': 'SGL', 'ss2': 'SimGCL'}
    try:
        conf = ModelConf('./config/' + models[num] + '.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
