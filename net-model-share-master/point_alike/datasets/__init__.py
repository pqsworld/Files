'''
Author: your name
Date: 2021-06-23 04:29:14
LastEditTime: 2021-07-07 06:39:18
LastEditors: your name
Description: In User Settings Edit
FilePath: /superpoint/datasets/__init__.py
'''
def get_dataset(name):
    mod = __import__('datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    return ''.join(n.capitalize() for n in name.split('_'))
