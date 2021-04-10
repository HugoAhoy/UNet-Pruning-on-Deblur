import os
import yaml
import shutil
import argparse


def loadYaml(yamlPath):
    print('LODING YAML')
    with open(yamlPath, 'r') as f:
        config = yaml.load(f)
        saveName = "%s-%s.yaml" % (config['note'], config['dataset'])
    if not os.path.exists('./config/' + saveName + '.yaml'):
        print('SAVE YAML FILE')
        shutil.copy(yamlPath, './config/' + saveName + '.yaml')
    else:
        print('YAML ALREADY EXIST!')
    return config, saveName


def parseArgs():
    parser = argparse.ArgumentParser(description="yaml file path")
    parser.add_argument('-c', '--config', default='yaml file path')
    args = parser.parse_args()
    return args
