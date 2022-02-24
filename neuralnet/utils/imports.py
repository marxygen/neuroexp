import importlib


def import_function(function_name: str, package_name: str):
    """Import a function from a package with specified name from a module with the name identical to the name of the function"""
    module = importlib.import_module(package_name + "." + function_name)
    return getattr(module, function_name)
