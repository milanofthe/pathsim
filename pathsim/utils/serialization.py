########################################################################################
##
##                   METHODS FOR SERIALIZATION OF PATHSIM OBJECTS
##                            (utils/serialization.py)
##
##                                Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import textwrap
import inspect
import types
import json
import ast
import re


# SERIALIZATION ========================================================================

def extract_source(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            # Extract the substring corresponding to the lambda expression
            lambda_source = source[node.col_offset:node.end_col_offset]
            return lambda_source
    return textwrap.dedent(source)
    

def serialize_callable(func):
    """Serialize a callable (function or lambda) 
    to a dictionary representation.
    
    Parameters
    ----------
    func : callable
        Function or lambda to serialize
    
    Returns
    -------
    out : dict
        Dictionary representation of the callable
    """
    # Get function name and determine if it's a lambda
    func_name = func.__name__
    is_lambda = func_name == '<lambda>'
    
    # Get source code
    try:
        # Try to get the source code
        source = extract_source(func)

        
        # Get function globals that are referenced in the function
        func_globals = {}
        if func.__globals__:

            # Get code object of the function
            code = func.__code__

            # Get names referenced in the function
            for name in code.co_names:
                if name in func.__globals__:
                    glob_value = func.__globals__[name]

                    # Only include serializable globals (modules and basic types)
                    if isinstance(glob_value, (int, float, str, bool, list, dict, tuple)) or \
                       inspect.ismodule(glob_value):
                        # For modules, just store the name
                        if inspect.ismodule(glob_value):
                            func_globals[name] = {
                                "type": "module",
                                "name": glob_value.__name__
                            }
                        else:
                            func_globals[name] = glob_value
        
        # Get function closures (important for nested functions)
        closures = {}
        if func.__closure__:
            closure_vars = inspect.getclosurevars(func)

            # Get nonlocal variables
            for name, value in closure_vars.nonlocals.items():

                # Only include serializable values
                if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                    closures[name] = value
        
        # Create the final representation
        return {
            "type": "lambda" if is_lambda else "function",
            "name": func_name,
            "source": source,
            "globals": func_globals,
            "closures": closures
        }
        
    except (IOError, TypeError) as e:

        # Fallback for dynamically created functions or builtins
        return {
            "type": "unserializable_callable",
            "name": func_name,
            "repr": repr(func)
        }


# DESERIALIZATION ======================================================================

def deserialize_callable(func_dict, global_env=None):
    """Deserialize a callable from its dictionary representation.
    
    Parameters
    ----------
    func_dict : dict
        Dictionary representation of the callable
    global_env : dict, optional
        Additional global environment to use when evaluating the function
    
    Returns
    -------
    func : callable
        Reconstructed function or lambda
    """
    if func_dict["type"] == "unserializable_callable":

        # Can't reconstruct the function, raise an error
        raise ValueError(f"Cannot deserialize function '{func_dict['name']}': {func_dict['repr']}")
    
    # Prepare environment for function evaluation
    if global_env is None:
        global_env = {}
    
    # Import needed modules
    for name, value in func_dict["globals"].items():
        if isinstance(value, dict) and value.get("type") == "module":
            try:
                global_env[name] = __import__(value["name"])
            except ImportError:
                pass  # Module can't be imported, ignore
        else:
            global_env[name] = value
    
    # Add closure variables to the environment
    for name, value in func_dict["closures"].items():
        global_env[name] = value
    
    # Handle different function types
    if func_dict["type"] == "lambda":

        # For lambdas, we can just eval the source
        source = func_dict["source"]
        try:
            return eval(source, global_env)
        except Exception as e:
            raise ValueError(f"Error deserializing lambda: {e}")
    else:

        # For regular functions, we need to exec the source and get the function from locals
        source = func_dict["source"]
        local_vars = {}
        try:

            # Execute the function definition
            exec(source, global_env, local_vars)

            # Get the function from local variables
            return local_vars[func_dict["name"]]
        except Exception as e:
            raise ValueError(f"Error deserializing function: {e}")



# CLASS FOR AUTOMATIC SERIALIZATION CAPABILITIES =======================================

class Serializable:
    """Mixin that provides automatic serialization based on __init__ parameters 
    and loading/saving to json formatted readable files

    """
    

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


    def save(self, path=""):
        """Save the dictionary representation of object to an external file
        
        Parameters
        ----------
        path : str
            filepath to save data to
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


    @classmethod
    def load(cls, path=""):
        """Load and instantiate an object from an external file in json format
        
        Parameters
        ----------
        path : str
            filepath to load data from

        Returns
        -------
        out : obj
            reconstructed object from dict representation
        """
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
        return None
        

    def to_dict(self):
        """Convert object to dictionary representation
        
        Returns
        -------
        result : dict
            representation of object
        """
        
        result = {
            "id": id(self),
            "type": self.__class__.__name__,
            "params": {}
        }
        
        # Get parameter names from __init__ signature
        signature = inspect.signature(self.__init__)
        param_names = [p for p in signature.parameters if p != 'self']
        
        # Get current values of parameters
        for name in param_names:
            if hasattr(self, name):
                value = getattr(self, name)
                # Handle callable parameters
                if callable(value):
                    result["params"][name] = serialize_callable(value)
                else:
                    # Try standard serialization
                    try:
                        json.dumps(value)
                        result["params"][name] = value
                    except (TypeError, OverflowError):
                        # For non-serializable objects, store their string representation
                        result["params"][name] = str(value)
            
        return result
    

    @classmethod
    def from_dict(cls, data):
        """Create block instance from dictionary representation.

        Parameters
        ----------
        data : dict
            representation of object

        Returns
        -------
        out : obj
            reconstructed object from dict representation            
        """

        # Use the class specified in the data
        block_type = data.get("type")
        
        # Find the class in the module hierarchy
        target_cls = cls._find_class(block_type)
        
        # If this is already the target class, or we couldn't find the target
        if target_cls is None or target_cls == cls:

            # Deserialize parameters
            params = {}
            for name, value in data["params"].items():
                if (isinstance(value, dict) and value.get("type") 
                	in ["lambda", "function", "unserializable_callable"]):
                    try:
                        params[name] = deserialize_callable(value)
                    except ValueError:

                        # Skip this parameter if we can't deserialize it
                        continue
                else:
                    params[name] = value
            
            # Create the instance
            return cls(**params)
        else:
            # Let the target class handle deserialization
            return target_cls.from_dict(data)


    @classmethod
    def _find_class(cls, class_name):
        """Find a class by name in the module hierarchy"""
        
        #first check if this is the class we're looking for
        if cls.__name__ == class_name:
            return cls
        
        #if not, check all subclasses recursively
        for subclass in cls.__subclasses__():
            if subclass.__name__ == class_name:
                return subclass
            
            #recursively check subclasses of this subclass
            found = subclass._find_class(class_name)
            if found:
                return found
        
        return None