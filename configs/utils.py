from ast import arg
import operator
import yaml

from functools import reduce

def args_to_dict(parser, args):

    d = dict({})

    for group in parser._action_groups:
        d[group.title] = {a.dest: getattr(
            args, a.dest, None) for a in group._group_actions}

    return d
    
def convert_yaml_dict_to_arg_list(yaml_dict):

    arg_list = list()
    print(yaml_dict)

    for k,v in yaml_dict.items():
        if v is None:
            continue
        for args, value in v.items():
            arg_list.append(args)
            if type(value) == list:
                arg_list.extend(value)
            elif value is not None:
                arg_list.append(value)

    print(arg_list)
    return arg_list

def read_yaml_file(file_path):
   
    with open(file_path) as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

    return yaml_dict


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def findDiff(d1, d2, path=[]):
    mismatch = list()
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                mismatch.extend(findDiff(d1[k],d2[k], path + [k]))
        else:
            mismatch.append(path + [k])
    
    return mismatch


def fill_dict_with_missing_value(existing_dict, new_dict, verbose=True):
    deprecated_key = findDiff(existing_dict, new_dict)
    
    if verbose and len(deprecated_key) != 0:
        print("Following keys are no longer in new_dict\n\t",["->".join(keys) for keys in deprecated_key])
    
    missing_new_key = findDiff(new_dict, existing_dict)
    
    if verbose and len(missing_new_key) != 0:
        print("Following key were missing and added to existing_dict\n\t", ["->".join(keys) for keys in missing_new_key])
    
    for key in missing_new_key:
        setInDict(existing_dict, key, getFromDict(new_dict, key))
        
    return existing_dict